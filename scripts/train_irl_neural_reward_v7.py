#!/usr/bin/env python3
"""
V7: Neural Reward IRL — practical Jetson version.

Single self-contained script: dataset generation → MLP training → testing → figures.
Scaled for Jetson ARM (no GPU, single process).

Key differences from V6 Heavy:
  - Neural network reward model instead of linear theta
  - Choice-based cross-entropy loss instead of max-entropy feature matching
  - Teacher utility function generates pseudo-labels for candidate paths
  - No expert_density dependency
"""

import os, sys, math, heapq, time, json, argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────
MAP_FILE = "outputs/las_vegas_big_map/las_vegas_big_map.npz"
OUT_DIR = "outputs/nuplan_irl_neural_reward_v7"

FEATURE_NAMES = [
    "length_ratio", "length_efficiency",
    "valid_ratio", "lane_ratio", "lane_connector_ratio", "intersection_ratio",
    "static_clearance_mean", "static_clearance_min",
    "obstacle_clearance_mean", "obstacle_clearance_min",
    "near_static_penalty", "near_obstacle_penalty",
    "heading_to_goal", "turn_mean", "turn_max",
    "goal_reached", "combined_clearance_min", "num_points_norm",
]

Point = Tuple[int, int]

# ── Map loading ─────────────────────────────────────────────────
def load_grid_npz(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    for key in ["grid", "map", "semantic_map", "arr_0"]:
        if key in data.files:
            arr = data[key]
            if arr.ndim == 2:
                return np.asarray(arr)
    for key in data.files:
        arr = data[key]
        if hasattr(arr, "ndim") and arr.ndim == 2:
            return np.asarray(arr)
    raise ValueError(f"No 2D array in {path}")

def decode_map(grid: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "valid": grid > 0,
        "lane": grid == 2,
        "lane_connector": grid == 3,
        "intersection": grid == 4,
    }

def build_context(map_path: str) -> Dict:
    grid = load_grid_npz(map_path)
    layers = decode_map(grid)
    valid = layers["valid"]
    h, w = grid.shape
    return {
        "grid": grid, "height": h, "width": w,
        "valid": valid,
        "drivable": valid.copy(),
        "lane": layers["lane"],
        "lane_connector": layers["lane_connector"],
        "intersection": layers["intersection"],
        "static_dist": distance_transform_edt(valid).astype(np.float32),
        "valid_coords": np.argwhere(valid).astype(np.int16),
    }

def inside(ctx, p):
    y, x = int(p[0]), int(p[1])
    return 0 <= y < ctx["height"] and 0 <= x < ctx["width"]

def euclidean(a, b):
    return float(math.hypot(float(a[0]-b[0]), float(a[1]-b[1])))

# ── Obstacle generation ────────────────────────────────────────
def clear_patch(mask, center, radius):
    h, w = mask.shape
    cy, cx = int(center[0]), int(center[1])
    y0, y1 = max(0, cy-radius), min(h, cy+radius+1)
    x0, x1 = max(0, cx-radius), min(w, cx+radius+1)
    mask[y0:y1, x0:x1] = False

def generate_obstacles(ctx, start, goal, rng, min_n, max_n):
    h, w = ctx["height"], ctx["width"]
    valid = ctx["valid"]
    valid_coords = ctx["valid_coords"]
    n_obs = int(rng.integers(min_n, max_n+1))
    mask = np.zeros((h, w), dtype=bool)
    sy, sx = start; gy, gx = goal
    dx, dy = float(gx-sx), float(gy-sy)
    norm = math.hypot(dx, dy) + 1e-6
    nx, ny = -dy/norm, dx/norm
    yy, xx = np.ogrid[:h, :w]

    for _ in range(n_obs):
        if rng.random() < 0.75:
            t = float(rng.uniform(0.08, 0.92))
            off = float(rng.normal(0.0, rng.uniform(3.0, 14.0)))
            cy = int(np.clip(round(sy + t*dy + off*ny), 0, h-1))
            cx = int(np.clip(round(sx + t*dx + off*nx), 0, w-1))
        else:
            cy, cx = valid_coords[int(rng.integers(0, len(valid_coords)))]
        if not valid[cy, cx]:
            continue
        ry = int(rng.integers(2, 8))
        rx = int(rng.integers(2, 8))
        ellipse = (((yy-cy)/max(1,ry))**2 + ((xx-cx)/max(1,rx))**2) <= 1.0
        mask |= ellipse & valid

    clear_patch(mask, start, 5)
    clear_patch(mask, goal, 5)
    return mask & valid

# ── A* path finding ────────────────────────────────────────────
def compute_obstacle_dist(obstacle_mask):
    if obstacle_mask is None or not obstacle_mask.any():
        return np.full(obstacle_mask.shape, 999.0, dtype=np.float32)
    return distance_transform_edt(~obstacle_mask).astype(np.float32)

def astar_path(ctx, start, goal, obstacle_mask,
               static_weight=0.0, obstacle_weight=0.0,
               lane_weight=0.0, intersection_weight=0.0,
               min_static_clearance=0.0, min_obstacle_clearance=0.0,
               max_expansions=60000):
    if not inside(ctx, start) or not inside(ctx, goal):
        return None
    valid = ctx["valid"]
    static_dist = ctx["static_dist"]
    obstacle_dist = compute_obstacle_dist(obstacle_mask)
    h, w = ctx["height"], ctx["width"]

    blocked = ~valid.copy()
    if obstacle_mask is not None:
        blocked |= obstacle_mask
    if min_static_clearance > 0:
        blocked |= static_dist < float(min_static_clearance)
    if min_obstacle_clearance > 0 and obstacle_dist is not None:
        blocked |= obstacle_dist < float(min_obstacle_clearance)

    sy, sx = start; gy, gx = goal
    blocked[sy, sx] = False; blocked[gy, gx] = False

    cost = np.ones((h, w), dtype=np.float32)
    cost += float(static_weight) * np.exp(-np.clip(static_dist, 0, 20)/2.5)
    if obstacle_dist is not None:
        # Wider decay (3.0 instead of 2.0): obstacle penalty extends further
        cost += float(obstacle_weight) * np.exp(-np.clip(obstacle_dist, 0, 20)/3.0)
    if lane_weight != 0:
        cost += float(lane_weight) * (~(ctx["lane"]|ctx["lane_connector"])).astype(np.float32)
    if intersection_weight != 0:
        cost += float(intersection_weight) * ctx["intersection"].astype(np.float32)
    cost[blocked] = np.inf

    def hfun(p):
        return math.hypot(float(p[0]-gy), float(p[1]-gx))

    neighbors = [(-1,0,1.),(1,0,1.),(0,-1,1.),(0,1,1.),
                 (-1,-1,math.sqrt(2)),(-1,1,math.sqrt(2)),
                 (1,-1,math.sqrt(2)),(1,1,math.sqrt(2))]

    open_h = [(hfun(start), 0.0, start)]
    parent = {}; gscore = {start: 0.0}; closed = set(); expansions = 0

    while open_h:
        _, gcur, cur = heapq.heappop(open_h)
        if cur in closed: continue
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]; path.append(cur)
            return path[::-1]
        closed.add(cur); expansions += 1
        if expansions > max_expansions: return None
        cy, cx = cur
        for dy, dx, mc in neighbors:
            ny, nx = cy+dy, cx+dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w: continue
            if blocked[ny, nx]: continue
            if not np.isfinite(cost[ny, nx]) or not np.isfinite(cost[cy, cx]): continue
            step = mc * 0.5 * (float(cost[cy,cx]) + float(cost[ny,nx]))
            ng = gcur + step
            npnt = (ny, nx)
            if ng < gscore.get(npnt, float("inf")):
                gscore[npnt] = ng; parent[npnt] = cur
                heapq.heappush(open_h, (ng + hfun(npnt), ng, npnt))
    return None

# ── Candidate path generation ──────────────────────────────────
def nearest_free_point(ctx, p, obstacle_mask, radius=10):
    py, px = int(round(p[0])), int(round(p[1]))
    valid = ctx["valid"]
    best, best_d = None, float("inf")
    for r in range(radius+1):
        y0, y1 = max(0, py-r), min(ctx["height"], py+r+1)
        x0, x1 = max(0, px-r), min(ctx["width"], px+r+1)
        for y in range(y0, y1):
            for x in range(x0, x1):
                if not valid[y, x]: continue
                if obstacle_mask is not None and obstacle_mask[y, x]: continue
                d = (y-py)**2 + (x-px)**2
                if d < best_d: best, best_d = (y, x), d
        if best is not None: return best
    return None

def dedupe_paths(paths):
    out, keys = [], set()
    for p in paths:
        if p is None or len(p)<2: continue
        sk = tuple(p[::max(1,len(p)//20)])
        fk = (p[0], p[-1], len(p), sk)
        if fk not in keys: keys.add(fk); out.append(p)
    return out

def path_length(path):
    if path is None or len(path)<2: return float("inf")
    return sum(math.hypot(float(a[0]-b[0]), float(a[1]-b[1])) for a,b in zip(path[:-1], path[1:]))

def generate_candidates(ctx, start, goal, obstacle_mask, count, rng, use_waypoints=False):
    modes = [
        # Shortest-like (no obstacle avoidance) — baseline
        {"static_weight":0.0, "obstacle_weight":0.0, "lane_weight":0.0, "intersection_weight":0.0, "min_static_clearance":0.0, "min_obstacle_clearance":0.0},
        # Obstacle-averse (soft cost only)
        {"static_weight":0.0, "obstacle_weight":8.0, "lane_weight":0.0, "intersection_weight":0.0, "min_static_clearance":0.0, "min_obstacle_clearance":0.0},
        # Moderate obstacle clearance (hard min=2)
        {"static_weight":1.0, "obstacle_weight":10.0, "lane_weight":0.0, "intersection_weight":0.0, "min_static_clearance":1.0, "min_obstacle_clearance":2.0},
        # Strong obstacle clearance (hard min=3)
        {"static_weight":2.0, "obstacle_weight":12.0, "lane_weight":0.0, "intersection_weight":0.0, "min_static_clearance":1.0, "min_obstacle_clearance":3.0},
        # Very strong obstacle clearance (hard min=4)
        {"static_weight":2.5, "obstacle_weight":15.0, "lane_weight":0.0, "intersection_weight":0.0, "min_static_clearance":2.0, "min_obstacle_clearance":4.0},
        # Balanced with lane preference
        {"static_weight":0.7, "obstacle_weight":8.0, "lane_weight":0.6, "intersection_weight":0.0, "min_static_clearance":0.0, "min_obstacle_clearance":2.0},
        # Safe with intersection avoidance
        {"static_weight":0.7, "obstacle_weight":10.0, "lane_weight":0.2, "intersection_weight":0.7, "min_static_clearance":1.0, "min_obstacle_clearance":2.0},
        # Balanced safe
        {"static_weight":1.2, "obstacle_weight":10.0, "lane_weight":0.4, "intersection_weight":0.2, "min_static_clearance":1.0, "min_obstacle_clearance":2.0},
    ]
    paths = []
    for m in modes:
        p = astar_path(ctx, start, goal, obstacle_mask, **m)
        if p: paths.append(p)

    # Waypoint-based variants (disabled by default for speed)
    if use_waypoints:
        sy, sx = start; gy, gx = goal
        dx, dy = float(gx-sx), float(gy-sy)
        norm = math.hypot(dx, dy)+1e-6; nx, ny = -dy/norm, dx/norm
        wp_specs = [{"static_weight":0.0,"obstacle_weight":0.0,"lane_weight":0.0,"intersection_weight":0.0,"min_static_clearance":0.0,"min_obstacle_clearance":0.0},
                    {"static_weight":1.0,"obstacle_weight":4.0,"lane_weight":0.2,"intersection_weight":0.0,"min_static_clearance":1.0,"min_obstacle_clearance":1.0}]
        wps = []
        for t in [0.30, 0.50, 0.70]:
            for off in [-15, 15]:
                px = sx + t*dx + off*nx; py = sy + t*dy + off*ny
                wp = nearest_free_point(ctx, (int(round(py)),int(round(px))), obstacle_mask, radius=10)
                if wp and euclidean(start,wp)>=8 and euclidean(goal,wp)>=8 and wp not in wps:
                    wps.append(wp)
        for wp in wps[:4]:
            for s in wp_specs:
                p1 = astar_path(ctx, start, wp, obstacle_mask, **s)
                p2 = astar_path(ctx, wp, goal, obstacle_mask, **s)
                if p1 and p2: paths.append(p1 + p2[1:])

    paths = dedupe_paths(paths)
    paths.sort(key=lambda p: path_length(p))
    return paths[:count]

# ── Feature extraction ─────────────────────────────────────────
def angle_wrap(a): return (a+np.pi)%(2.0*np.pi)-np.pi

def path_feature_vector(ctx, path, obstacle_mask, start, goal):
    if path is None or len(path)<2:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    obstacle_dist = compute_obstacle_dist(obstacle_mask)
    arr = np.asarray(path, dtype=np.int32)
    ys = np.clip(arr[:,0], 0, ctx["height"]-1); xs = np.clip(arr[:,1], 0, ctx["width"]-1)
    valid_v = ctx["valid"][ys,xs].astype(np.float32)
    lane_v = ctx["lane"][ys,xs].astype(np.float32)
    conn_v = ctx["lane_connector"][ys,xs].astype(np.float32)
    inter_v = ctx["intersection"][ys,xs].astype(np.float32)
    static_v = ctx["static_dist"][ys,xs].astype(np.float32)
    obs_v = obstacle_dist[ys,xs].astype(np.float32)
    length = path_length(path); straight = euclidean(start, goal)
    length_ratio = length/max(1.0, straight)
    length_efficiency = straight/max(1.0, length)
    static_mean = float(np.mean(np.clip(static_v,0,12))/12.0)
    static_min = float(np.min(np.clip(static_v,0,12))/12.0)
    obs_mean = float(np.mean(np.clip(obs_v,0,12))/12.0)
    obs_min = float(np.min(np.clip(obs_v,0,12))/12.0)
    near_static = float(np.mean(np.exp(-np.clip(static_v,0,20)/2.0)))
    near_obstacle = float(np.mean(np.exp(-np.clip(obs_v,0,20)/3.0)))  # wider penalty radius
    dy = np.diff(arr[:,0]).astype(np.float32); dx = np.diff(arr[:,1]).astype(np.float32)
    sn = np.sqrt(dx*dx+dy*dy)+1e-6
    gx, gy = float(goal[1]-start[1]), float(goal[0]-start[0])
    gnorm = math.hypot(gx,gy)+1e-6
    h2g = float(np.mean((dx/sn)*(gx/gnorm)+(dy/sn)*(gy/gnorm)))
    if len(dx)>=2:
        ang = np.arctan2(dy,dx); dt = np.abs(angle_wrap(np.diff(ang)))
        t_mean = float(np.mean(dt)/np.pi); t_max = float(np.max(dt)/np.pi)
    else:
        t_mean = t_max = 0.0
    end_dist = euclidean(path[-1], goal)
    goal_r = math.exp(-end_dist/2.0)
    ccm = min(static_min, obs_min)
    npn = min(1.0, len(path)/240.0)
    return np.array([length_ratio, length_efficiency,
        float(np.mean(valid_v)), float(np.mean(lane_v)),
        float(np.mean(conn_v)), float(np.mean(inter_v)),
        static_mean, static_min, obs_mean, obs_min,
        near_static, near_obstacle, h2g, t_mean, t_max,
        goal_r, ccm, npn], dtype=np.float32)

# ── Teacher utility ────────────────────────────────────────────
def teacher_utility(feat):
    f = {n: float(feat[i]) for i,n in enumerate(FEATURE_NAMES)}
    u = 0.0
    u += 10.0*f["goal_reached"] + 4.0*f["length_efficiency"]
    u -= 2.3*max(0.0, f["length_ratio"]-1.0)
    u += 1.2*f["lane_ratio"] + 0.4*f["lane_connector_ratio"] + 0.2*f["intersection_ratio"]
    u += 2.2*f["static_clearance_min"] + 1.1*f["static_clearance_mean"]
    # Strong obstacle avoidance: heavily reward min clearance, penalize proximity
    u += 12.0*f["obstacle_clearance_min"] + 5.0*f["obstacle_clearance_mean"]
    u += 8.0*f["combined_clearance_min"]
    u -= 4.0*f["near_static_penalty"] - 15.0*f["near_obstacle_penalty"]
    u += 1.6*f["heading_to_goal"] - 2.5*f["turn_mean"] - 1.2*f["turn_max"]
    return float(u)

# ── MLP Model ──────────────────────────────────────────────────
def make_model(fdim, hidden=256, dropout=0.10):
    return nn.Sequential(
        nn.Linear(fdim, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden//2), nn.ReLU(),
        nn.Linear(hidden//2, 1),
    )

# ── Dataset generation ─────────────────────────────────────────
def make_training_case(ctx, idx, seed, candidate_count, min_obs, max_obs,
                       min_dist, max_dist, max_tries=40):
    rng = np.random.default_rng(seed + idx*9973)
    vc = ctx["valid_coords"]
    for _ in range(max_tries):
        s = tuple(map(int, vc[int(rng.integers(0, len(vc)))]))
        g = tuple(map(int, vc[int(rng.integers(0, len(vc)))]))
        d = euclidean(s, g)
        if d < min_dist or d > max_dist: continue
        obs = generate_obstacles(ctx, s, g, rng, min_obs, max_obs)
        paths = generate_candidates(ctx, s, g, obs, candidate_count, rng)
        if len(paths) < 2: continue
        feats = [path_feature_vector(ctx, p, obs, s, g) for p in paths]
        utils = [teacher_utility(f) for f in feats]
        n = len(feats)
        feat_arr = np.zeros((candidate_count, len(FEATURE_NAMES)), dtype=np.float32)
        mask = np.zeros((candidate_count,), dtype=np.float32)
        feat_arr[:n] = np.stack(feats, axis=0); mask[:n] = 1.0
        label = int(np.argmax(np.asarray(utils, dtype=np.float32)))
        return {"features": feat_arr, "mask": mask, "label": label}
    return None

def generate_dataset(ctx, args):
    ds_path = os.path.join(args.output_dir, "dataset_v7.npz")
    if args.reuse and os.path.exists(ds_path):
        print(f"Reusing existing dataset: {ds_path}")
        d = np.load(ds_path, allow_pickle=True)
        return d["X"], d["labels"], d["mask"]

    print(f"Generating V7 neural reward dataset...")
    print(f"  target: {args.total_cases} cases, {args.candidates} candidates/case")
    print(f"  obstacles: {args.min_obs}-{args.max_obs}/case")
    print(f"  start-goal dist: {args.min_dist}-{args.max_dist} cells")

    X_list, y_list, m_list = [], [], []
    t0 = time.time(); last_r = 0; attempt = 0
    while len(X_list) < args.total_cases:
        need = args.total_cases - len(X_list)
        batch = max(32, min(need*3, 256))
        for i in range(attempt, attempt+batch):
            attempt += 1
            res = make_training_case(ctx, i, args.seed, args.candidates,
                                     args.min_obs, args.max_obs,
                                     args.min_dist, args.max_dist)
            if res is None: continue
            X_list.append(res["features"]); y_list.append(res["label"]); m_list.append(res["mask"])
            n = len(X_list)
            if n - last_r >= args.progress_every or n == args.total_cases:
                elapsed = time.time()-t0
                rate = n/max(1e-6, elapsed)
                eta = (args.total_cases-n)/max(1e-6, rate)
                print(f"  valid={n}/{args.total_cases}, attempts={attempt}, "
                      f"elapsed={elapsed:.0f}s, rate={rate:.1f}/s, eta={eta/60:.1f}min", flush=True)
                last_r = n
            if n >= args.total_cases: break

    X = np.stack(X_list, axis=0).astype(np.float32)
    labels = np.asarray(y_list, dtype=np.int64)
    mask = np.stack(m_list, axis=0).astype(np.float32)
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(ds_path, X=X, labels=labels, mask=mask, feature_names=np.array(FEATURE_NAMES))
    print(f"Dataset saved: {ds_path}  X={X.shape} labels={labels.shape} mask={mask.shape}")
    return X, labels, mask

# ── Training ───────────────────────────────────────────────────
def evaluate_model(model, loader, device):
    model.eval(); total_loss=0.0; total_c=0; total_correct=0
    with torch.no_grad():
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            b, c, fdim = xb.shape
            scores = model(xb.reshape(b*c, fdim)).reshape(b, c)
            scores = scores.masked_fill(mb <= 0.0, -1e9)
            loss = F.cross_entropy(scores, yb)
            pred = torch.argmax(scores, dim=1)
            total_correct += int((pred==yb).sum().item())
            total_loss += float(loss.item())*b; total_c += b
    return total_loss/max(1,total_c), total_correct/max(1,total_c)

def train_model(ctx, args, X, labels, mask):
    n = X.shape[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_train = int(n*0.70); n_val = int(n*0.15)
    train_idx = perm[:n_train]; val_idx = perm[n_train:n_train+n_val]
    test_idx = perm[n_train+n_val:]
    print(f"\nSplit: train={n_train}, val={n_val}, test={n-n_train-n_val}")

    # Standardize
    train_valid = X[train_idx][mask[train_idx]>0.0]
    feat_mean = train_valid.mean(axis=0).astype(np.float32)
    feat_std = np.maximum(train_valid.std(axis=0), 1e-6).astype(np.float32)
    X_std = ((X - feat_mean.reshape(1,1,-1)) / feat_std.reshape(1,1,-1)).astype(np.float32)

    device = torch.device("cpu")
    print(f"Training device: {device}")

    train_ds = TensorDataset(torch.from_numpy(X_std[train_idx]).float(),
                             torch.from_numpy(mask[train_idx]).float(),
                             torch.from_numpy(labels[train_idx]).long())
    val_ds = TensorDataset(torch.from_numpy(X_std[val_idx]).float(),
                           torch.from_numpy(mask[val_idx]).float(),
                           torch.from_numpy(labels[val_idx]).long())
    test_ds = TensorDataset(torch.from_numpy(X_std[test_idx]).float(),
                            torch.from_numpy(mask[test_idx]).float(),
                            torch.from_numpy(labels[test_idx]).long())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = make_model(len(FEATURE_NAMES), args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    best_val_loss = float("inf"); best_path = os.path.join(args.output_dir, "neural_reward_v7_best.pt")
    loss_rows = []

    print(f"Training {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}, hidden={args.hidden_dim}")
    for epoch in range(args.epochs):
        model.train(); epoch_loss=0.0; epoch_c=0; epoch_correct=0
        for xb, mb, yb in train_loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            b, c, fd = xb.shape
            scores = model(xb.reshape(b*c, fd)).reshape(b, c)
            scores = scores.masked_fill(mb <= 0.0, -1e9)
            loss = F.cross_entropy(scores, yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            pred = torch.argmax(scores.detach(), dim=1)
            epoch_correct += int((pred==yb).sum().item())
            epoch_loss += float(loss.item())*b; epoch_c += b
        scheduler.step()
        train_loss = epoch_loss/max(1,epoch_c)
        train_acc = epoch_correct/max(1,epoch_c)
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        loss_rows.append([epoch, train_loss, train_acc, val_loss, val_acc])
        if epoch%10==0 or epoch==args.epochs-1:
            print(f"  epoch={epoch:04d} train_loss={train_loss:.6f} acc={train_acc:.4f} "
                  f"val_loss={val_loss:.6f} acc={val_acc:.4f}", flush=True)
        ckpt = {"model_state": model.state_dict(), "feature_dim": len(FEATURE_NAMES),
                "hidden_dim": args.hidden_dim, "dropout": args.dropout,
                "feature_names": FEATURE_NAMES}
        torch.save(ckpt, os.path.join(args.output_dir, "neural_reward_v7_last.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss; torch.save(ckpt, best_path)

    # Test evaluation
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate_model(model, test_loader, device)
    print(f"\nTest: loss={test_loss:.6f} acc={test_acc:.4f}")

    # Save stats
    np.savez(os.path.join(args.output_dir, "feature_stats_v7.npz"),
             mean=feat_mean, std=feat_std, feature_names=np.array(FEATURE_NAMES))
    loss_arr = np.array(loss_rows)
    np.savetxt(os.path.join(args.output_dir, "training_loss_v7.csv"), loss_arr,
               delimiter=",", header="epoch,train_loss,train_acc,val_loss,val_acc", comments="")

    # Loss plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=140)
    ax1.plot(loss_arr[:,0], loss_arr[:,1], label="train loss"); ax1.plot(loss_arr[:,0], loss_arr[:,3], label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("cross entropy"); ax1.grid(True, alpha=0.3); ax1.legend()
    ax1.set_title("Neural Reward IRL V7 — Training Loss")
    ax2.plot(loss_arr[:,0], loss_arr[:,2], label="train acc"); ax2.plot(loss_arr[:,0], loss_arr[:,4], label="val acc")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.grid(True, alpha=0.3); ax2.legend()
    ax2.set_title("Choice Accuracy")
    fig.tight_layout()
    loss_png = os.path.join(args.output_dir, "neural_reward_training_v7.png")
    fig.savefig(loss_png, dpi=160); plt.close(fig)

    # Save config
    with open(os.path.join(args.output_dir, "model_config_v7.json"), "w") as f:
        json.dump({"version":"v7_neural_reward", "total_cases":int(n),
                   "split":{"train":n_train,"val":n_val,"test":n-n_train-n_val},
                   "candidates":args.candidates, "hidden_dim":args.hidden_dim,
                   "best_val_loss":float(best_val_loss), "test_loss":float(test_loss),
                   "test_acc":float(test_acc), "feature_names":FEATURE_NAMES}, f, indent=2)

    print(f"Saved: {best_path}, {loss_png}")
    return model, feat_mean, feat_std, device, {"test_loss": test_loss, "test_acc": test_acc}

# ── Testing / Planning ─────────────────────────────────────────
def score_paths(model, features, mean, std, device):
    x = ((features - mean.reshape(1,-1))/std.reshape(1,-1)).astype(np.float32)
    xt = torch.from_numpy(x).float().to(device)
    with torch.no_grad():
        return model(xt).reshape(-1).cpu().numpy()

def run_planning_tests(ctx, model, mean, std, device, args):
    fig_dir = os.path.join(args.output_dir, "test_figures")
    os.makedirs(fig_dir, exist_ok=True)
    rows = []; made = 0; attempt = 0
    print(f"\nRunning {args.num_tests} planning tests...")

    while made < args.num_tests and attempt < args.num_tests*50:
        seed = args.seed + 1000003*attempt; attempt += 1
        rng = np.random.default_rng(seed)
        vc = ctx["valid_coords"]
        s = tuple(map(int, vc[int(rng.integers(0, len(vc)))]))
        g = tuple(map(int, vc[int(rng.integers(0, len(vc)))]))
        d = euclidean(s, g)
        if d < args.min_dist or d > args.max_dist: continue
        obs = generate_obstacles(ctx, s, g, rng, args.min_obs, args.max_obs)
        paths = generate_candidates(ctx, s, g, obs, args.candidates, rng)
        if len(paths) < 2: continue
        feats = np.stack([path_feature_vector(ctx, p, obs, s, g) for p in paths], axis=0)

        scores = score_paths(model, feats, mean, std, device)
        best_idx = int(np.argmax(scores))
        irl_path = paths[best_idx]

        # Also get shortest path as baseline
        short = astar_path(ctx, s, g, obs, static_weight=0, obstacle_weight=0)

        # Metrics
        obs_dist = compute_obstacle_dist(obs)
        ip = np.asarray(irl_path, dtype=np.int32)
        iys = np.clip(ip[:,0],0,ctx["height"]-1); ixs = np.clip(ip[:,1],0,ctx["width"]-1)
        irl_len = path_length(irl_path)
        irl_static_min = float(np.min(ctx["static_dist"][iys, ixs]))
        irl_obst_min = float(np.min(obs_dist[iys, ixs]))
        success = int(euclidean(irl_path[-1], g) <= 2.0)

        if short and len(short)>1:
            sp = np.asarray(short, dtype=np.int32)
            sys_ = np.clip(sp[:,0],0,ctx["height"]-1); sxs_ = np.clip(sp[:,1],0,ctx["width"]-1)
            short_len = path_length(short)
            short_static_min = float(np.min(ctx["static_dist"][sys_, sxs_]))
            short_obst_min = float(np.min(obs_dist[sys_, sxs_]))
        else:
            short_len = irl_len; short_static_min = irl_static_min; short_obst_min = irl_obst_min

        rows.append({"id":made, "success":success,
                     "irl_len":irl_len, "short_len":short_len,
                     "len_ratio":irl_len/max(1e-6, short_len),
                     "irl_static_min":irl_static_min, "irl_obst_min":irl_obst_min,
                     "short_static_min":short_static_min, "short_obst_min":short_obst_min})

        # Plot first 12 cases
        if made < 12:
            title = f"V7 Neural IRL Test #{made} | IRL={irl_len:.0f} Short={short_len:.0f}"
            fig_path = os.path.join(fig_dir, f"test_{made:03d}.png")
            _plot_case(ctx, obs, s, g, irl_path, short, fig_path, title)

        print(f"  test={made:03d} succ={success} irl_len={irl_len:.1f} short_len={short_len:.1f} "
              f"ratio={irl_len/max(1e-6,short_len):.3f} sta_min={irl_static_min:.1f} obs_min={irl_obst_min:.1f}", flush=True)
        made += 1

    # Summary
    sr = np.mean([r["success"] for r in rows])
    mil = np.mean([r["irl_len"] for r in rows])
    msl = np.mean([r["short_len"] for r in rows])
    mr = np.mean([r["len_ratio"] for r in rows])
    print(f"\nSummary: success={sr:.3f} irl_len={mil:.1f} short_len={msl:.1f} ratio={mr:.3f}")

    # Summary figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    ids = [r["id"] for r in rows]
    axes[0,0].bar(ids, [r["irl_len"] for r in rows], alpha=0.7, label="IRL")
    axes[0,0].bar(ids, [r["short_len"] for r in rows], alpha=0.7, label="Shortest")
    axes[0,0].legend(); axes[0,0].set_title("Path Length"); axes[0,0].set_xlabel("test id")
    axes[0,1].scatter(ids, [r["len_ratio"] for r in rows]); axes[0,1].axhline(1.0, color='r', ls='--')
    axes[0,1].set_title("Length Ratio (IRL/Short)"); axes[0,1].set_xlabel("test id")
    axes[1,0].bar(ids, [r["irl_static_min"] for r in rows], alpha=0.7, label="IRL")
    axes[1,0].bar(ids, [r["short_static_min"] for r in rows], alpha=0.7, label="Shortest")
    axes[1,0].legend(); axes[1,0].set_title("Min Static Clearance"); axes[1,0].set_xlabel("test id")
    axes[1,1].bar(ids, [r["irl_obst_min"] for r in rows], alpha=0.7, label="IRL")
    axes[1,1].bar(ids, [r["short_obst_min"] for r in rows], alpha=0.7, label="Shortest")
    axes[1,1].legend(); axes[1,1].set_title("Min Obstacle Clearance"); axes[1,1].set_xlabel("test id")
    fig.suptitle(f"V7 Neural Reward IRL Planning Results (success={sr:.2f}, ratio={mr:.3f})", fontweight="bold")
    fig.tight_layout()
    sum_png = os.path.join(args.output_dir, "planning_summary_v7.png")
    fig.savefig(sum_png, dpi=160); plt.close(fig)
    return rows

def _plot_case(ctx, obs, start, goal, irl_path, short_path, out_path, title):
    h, w = ctx["height"], ctx["width"]
    img = np.zeros((h,w,3), dtype=np.float32)+0.88
    img[ctx["valid"]] = [0.98,0.90,0.32]
    img[ctx["lane"]] = [0.40,0.75,0.80]
    img[ctx["lane_connector"]] = [0.98,0.80,0.25]
    img[ctx["intersection"]] = [1.00,0.84,0.30]
    if obs is not None: img[obs] = [0.02,0.02,0.02]
    fig, ax = plt.subplots(figsize=(8,8), dpi=130)
    ax.imshow(img, origin="upper", interpolation="nearest")
    if short_path and len(short_path)>1:
        sp = np.asarray(short_path); ax.plot(sp[:,1], sp[:,0], "b--", lw=2.2, label="shortest")
    if irl_path and len(irl_path)>1:
        ip = np.asarray(irl_path); ax.plot(ip[:,1], ip[:,0], "r-", lw=2.6, label="neural IRL")
    ax.scatter([start[1]],[start[0]], marker="s", s=80, c="black", label="start", zorder=5)
    ax.scatter([goal[1]],[goal[0]], marker="*", s=180, c="green", label="goal", zorder=6)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title); ax.legend(loc="best")
    ax.set_xlim(max(0,min(start[1],goal[1])-40), min(w-1,max(start[1],goal[1])+40))
    ax.set_ylim(min(h-1,max(start[0],goal[0])+40), max(0,min(start[0],goal[0])-40))
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)

# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default=MAP_FILE)
    parser.add_argument("--output-dir", default=OUT_DIR)
    parser.add_argument("--reuse", action="store_true", help="Reuse existing dataset")
    # Data
    parser.add_argument("--total-cases", type=int, default=800)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--min-obs", type=int, default=4)
    parser.add_argument("--max-obs", type=int, default=10)
    parser.add_argument("--min-dist", type=float, default=55)
    parser.add_argument("--max-dist", type=float, default=210)
    # Training
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    # Testing
    parser.add_argument("--num-tests", type=int, default=30)
    # Misc
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("V7: Neural Reward IRL — Practical Jetson Version")
    print("="*60)

    ctx = build_context(args.map)
    uniq, cnt = np.unique(ctx["grid"], return_counts=True)
    print(f"Map: {args.map} shape={ctx['grid'].shape}")
    for u, c in zip(uniq, cnt):
        print(f"  value {u}: {c}")

    # Dataset
    X, labels, mask = generate_dataset(ctx, args)

    # Train
    if not args.skip_train:
        model, mean, std, device, train_metrics = train_model(ctx, args, X, labels, mask)
    else:
        best_path = os.path.join(args.output_dir, "neural_reward_v7_best.pt")
        stats_path = os.path.join(args.output_dir, "feature_stats_v7.npz")
        ckpt = torch.load(best_path, map_location="cpu")
        model = make_model(int(ckpt["feature_dim"]), int(ckpt["hidden_dim"]), float(ckpt["dropout"]))
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        stats = np.load(stats_path)
        mean, std = stats["mean"].astype(np.float32), stats["std"].astype(np.float32)
        device = torch.device("cpu")
        print(f"Loaded model from {best_path}")

    # Test
    if not args.skip_test:
        rows = run_planning_tests(ctx, model, mean, std, device, args)
        # Print final summary
        sr = np.mean([r["success"] for r in rows])
        print(f"\nFinal: success_rate={sr:.3f} | mean_irl_len={np.mean([r['irl_len'] for r in rows]):.1f} "
              f"mean_short_len={np.mean([r['short_len'] for r in rows]):.1f} "
              f"mean_ratio={np.mean([r['len_ratio'] for r in rows]):.3f}")

    print("\nDone. Outputs in:", args.output_dir)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3

# -*- coding: utf-8 -*-



import heapq

import math

from pathlib import Path



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy.ndimage import distance_transform_edt, binary_dilation





MAP_PATH = Path("outputs/las_vegas_big_map/las_vegas_big_map.npz")

DEMO_PATH = Path("outputs/las_vegas_big_map/multi_expert_trajectories_grid.csv")

OUT_DIR = Path("outputs/las_vegas_big_map_irl")



STATE_RADIUS = 35      # cells around expert trajectories

EPOCHS = 60

LR = 0.10

GAMMA = 0.95

BETA = 3.0

HORIZON = 140



ACTIONS = [

    (1, 0), (-1, 0), (0, 1), (0, -1),

    (1, 1), (1, -1), (-1, 1), (-1, -1),

]





def load_data():

    mp = np.load(MAP_PATH)

    grid = mp["grid"]

    demos = pd.read_csv(DEMO_PATH)

    return grid, demos





def build_state_mask(grid, demos):

    drivable = grid > 0



    demo_mask = np.zeros_like(grid, dtype=bool)

    for _, r in demos.iterrows():

        x, y = int(r["gx"]), int(r["gy"])

        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:

            demo_mask[y, x] = True



    # Expand around demonstrations to form a large but bounded learning area

    mask = binary_dilation(demo_mask, iterations=STATE_RADIUS)

    mask = mask & drivable



    return mask





def valid_state(mask, s):

    x, y = s

    if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:

        return False

    return bool(mask[y, x])





def extract_demo_paths(demos, mask):

    paths = []



    for demo_id, g in demos.groupby("demo_id"):

        g = g.sort_values("t")

        raw = [(int(r["gx"]), int(r["gy"])) for _, r in g.iterrows()]



        path = []

        for p in raw:

            if valid_state(mask, p):

                if not path or p != path[-1]:

                    path.append(p)



        if len(path) >= 10:

            paths.append(path)



    return paths





def build_states(mask):

    ys, xs = np.where(mask)

    states = list(zip(xs.tolist(), ys.tolist()))

    s2i = {s: i for i, s in enumerate(states)}

    return states, s2i





def compute_static_features(grid, mask, states, demo_paths):

    # Semantic features

    # grid labels:

    # 1 generic drivable, 2 lane, 3 connector, 4 intersection



    lane = grid == 2

    connector = grid == 3

    intersection = grid == 4



    demo_mask = np.zeros_like(grid, dtype=bool)

    for path in demo_paths:

        for x, y in path:

            demo_mask[y, x] = True



    # Distance to any expert trajectory, normalized

    dist_to_demo = distance_transform_edt(~demo_mask)



    feats = []



    for x, y in states:

        f_lane = 1.0 if lane[y, x] else 0.0

        f_connector = 1.0 if connector[y, x] else 0.0

        f_intersection = 1.0 if intersection[y, x] else 0.0

        f_demo_dist = -min(float(dist_to_demo[y, x]), 50.0) / 50.0



        feats.append([f_lane, f_connector, f_intersection, f_demo_dist])



    return np.asarray(feats, dtype=np.float64)





def goal_feature(states, goal, grid_shape):

    gx, gy = goal

    max_d = math.hypot(grid_shape[1], grid_shape[0])

    return np.asarray(

        [-math.hypot(x - gx, y - gy) / max_d for x, y in states],

        dtype=np.float64,

    ).reshape(-1, 1)





class MultiDemoIRL:

    def __init__(self, grid, mask, demo_paths):

        self.grid = grid

        self.mask = mask

        self.demo_paths = demo_paths



        self.states, self.s2i = build_states(mask)

        self.static_features = compute_static_features(grid, mask, self.states, demo_paths)



        # weights:

        # lane, connector, intersection, demo_density, goal_progress

        self.theta = np.zeros(5, dtype=np.float64)



    def transition(self, state, action):

        ns = (state[0] + action[0], state[1] + action[1])

        if valid_state(self.mask, ns):

            return ns

        return state



    def features(self, goal):

        return np.hstack([self.static_features, goal_feature(self.states, goal, self.grid.shape)])



    def reward(self, goal):

        return self.features(goal) @ self.theta



    def soft_value_iteration(self, goal):

        rewards = self.reward(goal)

        values = np.zeros(len(self.states), dtype=np.float64)



        for _ in range(50):

            new_values = np.zeros_like(values)



            for i, state in enumerate(self.states):

                qs = []

                for action in ACTIONS:

                    ns = self.transition(state, action)

                    j = self.s2i[ns]

                    qs.append(rewards[j] + GAMMA * values[j])



                qs = np.asarray(qs)

                z = BETA * qs

                m = np.max(z)

                new_values[i] = (m + np.log(np.sum(np.exp(z - m)))) / BETA



            if np.max(np.abs(new_values - values)) < 1e-5:

                values = new_values

                break



            values = new_values



        policy = np.zeros((len(self.states), len(ACTIONS)), dtype=np.float64)



        for i, state in enumerate(self.states):

            qs = []

            for action in ACTIONS:

                ns = self.transition(state, action)

                j = self.s2i[ns]

                qs.append(rewards[j] + GAMMA * values[j])



            qs = np.asarray(qs)

            z = BETA * qs

            z -= np.max(z)

            p = np.exp(z)

            p /= np.sum(p)

            policy[i] = p



        return policy



    def expert_mu(self, path, goal):

        feats = self.features(goal)

        mu = np.zeros(feats.shape[1], dtype=np.float64)

        count = 0



        for s in path:

            if s in self.s2i:

                mu += feats[self.s2i[s]]

                count += 1



        return mu / max(count, 1)



    def expected_svf(self, policy, start):

        d = np.zeros((HORIZON, len(self.states)), dtype=np.float64)

        d[0, self.s2i[start]] = 1.0



        for t in range(HORIZON - 1):

            for i, state in enumerate(self.states):

                if d[t, i] <= 0:

                    continue



                for a_idx, action in enumerate(ACTIONS):

                    ns = self.transition(state, action)

                    j = self.s2i[ns]

                    d[t + 1, j] += d[t, i] * policy[i, a_idx]



        return d.sum(axis=0) / HORIZON



    def train(self):

        rng = np.random.default_rng(3)

        self.theta = rng.normal(0, 0.03, size=5)



        losses = []



        for epoch in range(EPOCHS):

            grad_total = np.zeros_like(self.theta)

            loss_total = 0.0

            used = 0



            for path in self.demo_paths:

                start = path[0]

                goal = path[-1]



                if start not in self.s2i or goal not in self.s2i:

                    continue



                feats = self.features(goal)



                policy = self.soft_value_iteration(goal)

                expert_mu = self.expert_mu(path, goal)

                svf = self.expected_svf(policy, start)

                model_mu = svf @ feats



                grad = expert_mu - model_mu

                grad_total += grad

                loss_total += float(np.linalg.norm(grad))

                used += 1



            if used == 0:

                raise RuntimeError("No usable demo paths for training.")



            grad_total /= used

            loss = loss_total / used



            self.theta += LR * grad_total

            losses.append(loss)



            if epoch % 5 == 0 or epoch == EPOCHS - 1:

                print(f"epoch={epoch:03d}, loss={loss:.6f}, theta={np.round(self.theta, 3)}")



        return losses



    def astar_path(self, start, goal):

        reward = self.reward(goal)

        r_min, r_max = float(np.min(reward)), float(np.max(reward))

        r_norm = (reward - r_min) / max(r_max - r_min, 1e-6)



        def heuristic(s):

            return math.hypot(s[0] - goal[0], s[1] - goal[1])



        open_set = []

        heapq.heappush(open_set, (0.0, start))

        came_from = {}

        g_score = {start: 0.0}



        while open_set:

            _, current = heapq.heappop(open_set)



            if current == goal:

                break



            for action in ACTIONS:

                ns = self.transition(current, action)

                if ns == current:

                    continue



                j = self.s2i[ns]

                step_len = math.hypot(action[0], action[1])



                # High learned reward means lower path cost

                cost = step_len + 1.2 * (1.0 - r_norm[j])



                tentative = g_score[current] + cost



                if tentative < g_score.get(ns, float("inf")):

                    came_from[ns] = current

                    g_score[ns] = tentative

                    f = tentative + 0.8 * heuristic(ns)

                    heapq.heappush(open_set, (f, ns))



        if goal not in came_from and goal != start:

            print("Warning: A* did not reach goal.")

            return [start]



        path = [goal]

        cur = goal

        while cur != start:

            cur = came_from[cur]

            path.append(cur)



        path.reverse()

        return path





def evaluate(generated, expert):

    exp = np.asarray(expert, dtype=float)

    dists = []

    for p in generated:

        d = np.sqrt((exp[:, 0] - p[0]) ** 2 + (exp[:, 1] - p[1]) ** 2)

        dists.append(float(np.min(d)))



    fde = math.hypot(generated[-1][0] - expert[-1][0], generated[-1][1] - expert[-1][1])



    return {

        "mean_distance_to_expert": float(np.mean(dists)),

        "FDE": float(fde),

        "generated_length": len(generated),

        "expert_length": len(expert),

        "success": 1 if generated[-1] == expert[-1] else 0,

    }





def plot_results(grid, mask, demo_paths, test_path, generated, losses, theta):

    OUT_DIR.mkdir(parents=True, exist_ok=True)



    plt.figure(figsize=(12, 10))



    vis = np.zeros_like(grid, dtype=np.uint8)

    vis[grid > 0] = grid[grid > 0]

    vis[~mask] = 0



    plt.imshow(vis, origin="upper", alpha=0.9)



    for i, path in enumerate(demo_paths):

        arr = np.asarray(path)

        plt.plot(arr[:, 0], arr[:, 1], linewidth=1.0, alpha=0.45)



    tp = np.asarray(test_path)

    gp = np.asarray(generated)



    plt.plot(tp[:, 0], tp[:, 1], linewidth=2.5, label="held-out expert trajectory")

    plt.plot(gp[:, 0], gp[:, 1], linewidth=2.5, label="IRL generated path")



    plt.scatter([test_path[0][0]], [test_path[0][1]], marker="s", s=80, label="start")

    plt.scatter([test_path[-1][0]], [test_path[-1][1]], marker="*", s=150, label="goal")



    plt.title("Big Las Vegas semantic map: multi-demo MaxEnt IRL")

    plt.xlabel("grid x")

    plt.ylabel("grid y")

    plt.legend()

    plt.tight_layout()

    plt.savefig(OUT_DIR / "big_map_multi_demo_irl_path.png", dpi=180)

    plt.close()



    plt.figure(figsize=(8, 4))

    plt.plot(losses)

    plt.title("Multi-demo MaxEnt IRL training curve")

    plt.xlabel("epoch")

    plt.ylabel("mean feature expectation gap")

    plt.tight_layout()

    plt.savefig(OUT_DIR / "big_map_multi_demo_irl_loss.png", dpi=180)

    plt.close()



    with open(OUT_DIR / "learned_theta.txt", "w", encoding="utf-8") as f:

        names = ["lane", "lane_connector", "intersection", "demo_density", "goal_progress"]

        for n, v in zip(names, theta):

            f.write(f"{n}: {v:.6f}\n")





def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)



    grid, demos = load_data()

    mask = build_state_mask(grid, demos)

    demo_paths = extract_demo_paths(demos, mask)



    print(f"Grid shape: {grid.shape}")

    print(f"Valid learning states: {int(mask.sum())}")

    print(f"Expert demos: {len(demo_paths)}")



    if len(demo_paths) < 2:

        raise RuntimeError("Need at least 2 expert demos.")



    # Last demo used as held-out test path

    train_paths = demo_paths[:-1]

    test_path = demo_paths[-1]



    irl = MultiDemoIRL(grid, mask, train_paths)

    losses = irl.train()



    generated = irl.astar_path(test_path[0], test_path[-1])

    metrics = evaluate(generated, test_path)



    print("\nLearned theta:")

    names = ["lane", "lane_connector", "intersection", "demo_density", "goal_progress"]

    for n, v in zip(names, irl.theta):

        print(f"{n:20s}: {v:.6f}")



    print("\nEvaluation:")

    for k, v in metrics.items():

        print(f"{k:28s}: {v}")



    pd.DataFrame([metrics]).to_csv(OUT_DIR / "evaluation_metrics.csv", index=False)

    plot_results(grid, mask, demo_paths, test_path, generated, losses, irl.theta)



    print("\nSaved:")

    print(OUT_DIR / "big_map_multi_demo_irl_path.png")

    print(OUT_DIR / "big_map_multi_demo_irl_loss.png")

    print(OUT_DIR / "evaluation_metrics.csv")

    print(OUT_DIR / "learned_theta.txt")





if __name__ == "__main__":

    main()


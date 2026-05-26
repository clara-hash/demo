
#!/usr/bin/env python3

# -*- coding: utf-8 -*-



"""

Use extracted real nuPlan ego trajectory as expert demonstrations for MaxEnt IRL.



Input:

    outputs/nuplan_real/ego_trajectory.csv



Output:

    outputs/nuplan_real/real_irl_path.png

    outputs/nuplan_real/real_irl_loss.png

    outputs/nuplan_real/real_irl_log.txt



Run:

    python scripts/real_nuplan_irl_from_csv.py | tee outputs/nuplan_real/real_irl_log.txt

"""



import math

from pathlib import Path



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd





class Config:

    csv_path = Path("outputs/nuplan_real/ego_trajectory.csv")

    out_dir = Path("outputs/nuplan_real")



    # 只取一小段真实轨迹，避免地图太大、训练太慢

    start_index = 0

    num_poses = 700

    stride = 8



    resolution = 1.0       # 1 meter per cell

    margin = 15            # grid margin cells

    corridor_radius = 5    # drivable corridor radius around expert path



    gamma = 0.95

    beta = 4.0

    lr = 0.12

    epochs = 80

    horizon = 120





ACTIONS = [

    (1, 0), (-1, 0), (0, 1), (0, -1),

    (1, 1), (1, -1), (-1, 1), (-1, -1),

    (0, 0),

]





def bresenham(p0, p1):

    """Return integer grid cells between two cells."""

    x0, y0 = p0

    x1, y1 = p1



    cells = []

    dx = abs(x1 - x0)

    dy = -abs(y1 - y0)



    sx = 1 if x0 < x1 else -1

    sy = 1 if y0 < y1 else -1



    err = dx + dy

    x, y = x0, y0



    while True:

        cells.append((x, y))

        if x == x1 and y == y1:

            break



        e2 = 2 * err

        if e2 >= dy:

            err += dy

            x += sx

        if e2 <= dx:

            err += dx

            y += sy



    return cells





def load_expert_path(cfg):

    if not cfg.csv_path.exists():

        raise FileNotFoundError(cfg.csv_path)



    df = pd.read_csv(cfg.csv_path)

    df = df.iloc[cfg.start_index: cfg.start_index + cfg.num_poses: cfg.stride].copy()



    if len(df) < 5:

        raise RuntimeError("Too few poses. Try increasing num_poses.")



    xy = df[["rel_x", "rel_y"]].to_numpy(dtype=np.float64)



    # 平移到正坐标，转栅格

    min_xy = xy.min(axis=0)

    xy_shifted = xy - min_xy + cfg.margin * cfg.resolution



    grid_xy = np.round(xy_shifted / cfg.resolution).astype(int)



    # 用 Bresenham 补齐轨迹栅格，使相邻点连续

    dense_path = []

    for i in range(len(grid_xy) - 1):

        seg = bresenham(tuple(grid_xy[i]), tuple(grid_xy[i + 1]))

        if dense_path:

            dense_path.extend(seg[1:])

        else:

            dense_path.extend(seg)



    # 去除连续重复

    cleaned = []

    for p in dense_path:

        if not cleaned or p != cleaned[-1]:

            cleaned.append(p)



    return cleaned, xy





def build_grid_from_expert_path(path, cfg):

    xs = [p[0] for p in path]

    ys = [p[1] for p in path]



    width = max(xs) + cfg.margin + 1

    height = max(ys) + cfg.margin + 1



    # 0: non-drivable, 1: drivable corridor, 2: expert centerline, 4: goal

    grid = np.zeros((height, width), dtype=np.int32)



    for x, y in path:

        for dy in range(-cfg.corridor_radius, cfg.corridor_radius + 1):

            for dx in range(-cfg.corridor_radius, cfg.corridor_radius + 1):

                nx, ny = x + dx, y + dy

                if 0 <= nx < width and 0 <= ny < height:

                    if math.hypot(dx, dy) <= cfg.corridor_radius:

                        grid[ny, nx] = 1



    for x, y in path:

        if 0 <= x < width and 0 <= y < height:

            grid[y, x] = 2



    gx, gy = path[-1]

    grid[gy, gx] = 4



    return grid





def valid_state(grid, s):

    x, y = s

    if x < 0 or x >= grid.shape[1] or y < 0 or y >= grid.shape[0]:

        return False

    return grid[y, x] in [1, 2, 4]





def get_states(grid):

    states = []

    for y in range(grid.shape[0]):

        for x in range(grid.shape[1]):

            if valid_state(grid, (x, y)):

                states.append((x, y))

    return states





def nearest_centerline_distance(centerline_cells, s):

    x, y = s

    arr = np.asarray(centerline_cells, dtype=np.float64)

    d = np.sqrt((arr[:, 0] - x) ** 2 + (arr[:, 1] - y) ** 2)

    return float(np.min(d))





def features_for_state(grid, s, goal, centerline_cells):

    x, y = s

    gx, gy = goal



    max_goal_dist = math.hypot(grid.shape[1], grid.shape[0])

    goal_dist = math.hypot(gx - x, gy - y)



    center_dist = nearest_centerline_distance(centerline_cells, s)

    is_centerline = 1.0 if grid[y, x] == 2 else 0.0

    is_goal = 1.0 if s == goal else 0.0



    # 真实 nuPlan 轨迹版特征：

    # 1 接近目标，2 靠近专家轨迹中心线，3 位于专家轨迹中心线，4 到达目标

    return np.array([

        -goal_dist / max_goal_dist,

        -center_dist / 10.0,

        is_centerline,

        is_goal,

    ], dtype=np.float64)





def expert_feature_mean(path, state_to_index, features):

    mu = np.zeros(features.shape[1], dtype=np.float64)

    count = 0



    for s in path:

        if s in state_to_index:

            mu += features[state_to_index[s]]

            count += 1



    return mu / max(count, 1)





class MaxEntIRL:

    def __init__(self, grid, expert_path, cfg):

        self.grid = grid

        self.expert_path = expert_path

        self.cfg = cfg



        self.start = expert_path[0]

        self.goal = expert_path[-1]

        self.centerline_cells = sorted(set(expert_path))



        self.states = get_states(grid)

        self.state_to_index = {s: i for i, s in enumerate(self.states)}



        self.features = np.vstack([

            features_for_state(grid, s, self.goal, self.centerline_cells)

            for s in self.states

        ])



        self.theta = np.zeros(self.features.shape[1], dtype=np.float64)



    def transition(self, state, action):

        ns = (state[0] + action[0], state[1] + action[1])

        if valid_state(self.grid, ns):

            return ns

        return state



    def reward(self):

        return self.features @ self.theta



    def soft_value_iteration(self):

        rewards = self.reward()

        values = np.zeros(len(self.states), dtype=np.float64)



        for _ in range(60):

            new_values = np.zeros_like(values)



            for i, state in enumerate(self.states):

                q_values = []

                for action in ACTIONS:

                    ns = self.transition(state, action)

                    j = self.state_to_index[ns]

                    q_values.append(rewards[j] + self.cfg.gamma * values[j])



                q_values = np.asarray(q_values)

                z = self.cfg.beta * q_values

                m = np.max(z)

                new_values[i] = (m + np.log(np.sum(np.exp(z - m)))) / self.cfg.beta



            if np.max(np.abs(new_values - values)) < 1e-5:

                values = new_values

                break



            values = new_values



        policy = np.zeros((len(self.states), len(ACTIONS)), dtype=np.float64)



        for i, state in enumerate(self.states):

            q_values = []

            for action in ACTIONS:

                ns = self.transition(state, action)

                j = self.state_to_index[ns]

                q_values.append(rewards[j] + self.cfg.gamma * values[j])



            q_values = np.asarray(q_values)

            z = self.cfg.beta * q_values

            z -= np.max(z)

            probs = np.exp(z)

            probs /= np.sum(probs)

            policy[i] = probs



        return policy



    def expected_svf(self, policy):

        d = np.zeros((self.cfg.horizon, len(self.states)), dtype=np.float64)

        d[0, self.state_to_index[self.start]] = 1.0



        for t in range(self.cfg.horizon - 1):

            for i, state in enumerate(self.states):

                if d[t, i] <= 0:

                    continue



                for a_idx, action in enumerate(ACTIONS):

                    ns = self.transition(state, action)

                    j = self.state_to_index[ns]

                    d[t + 1, j] += d[t, i] * policy[i, a_idx]



        return d.sum(axis=0) / self.cfg.horizon



    def train(self):

        rng = np.random.default_rng(1)

        self.theta = rng.normal(0.0, 0.05, size=self.features.shape[1])



        expert_mu = expert_feature_mean(

            self.expert_path,

            self.state_to_index,

            self.features,

        )



        losses = []



        for epoch in range(self.cfg.epochs):

            policy = self.soft_value_iteration()

            svf = self.expected_svf(policy)

            model_mu = svf @ self.features



            grad = expert_mu - model_mu

            self.theta += self.cfg.lr * grad



            loss = float(np.linalg.norm(grad))

            losses.append(loss)



            if epoch % 10 == 0 or epoch == self.cfg.epochs - 1:

                print(

                    f"epoch={epoch:03d}, "

                    f"feature_gap={loss:.6f}, "

                    f"theta={np.round(self.theta, 3)}"

                )



        return losses



    def _nearest_progress_index(self, state):

        """Return the nearest index on the expert trajectory.



        This is used to impose a forward-progress constraint during

        trajectory generation. Without this constraint, the local grid

        corridor is bidirectional and the generated path may loop back.

        """

        arr = np.asarray(self.expert_path, dtype=np.float64)

        x, y = state

        d = np.sqrt((arr[:, 0] - x) ** 2 + (arr[:, 1] - y) ** 2)

        return int(np.argmin(d))



    def generate_path(self, max_steps=300):

        """Generate a path with a forward-progress constraint.



        The MaxEnt IRL policy gives action probabilities, while this

        function adds a trajectory-prediction constraint: the path should

        move forward along the expert-trajectory index and avoid revisits.

        """

        policy = self.soft_value_iteration()



        path = [self.start]

        state = self.start

        visited = {state}



        for _ in range(max_steps):

            if state == self.goal:

                break



            current_progress = self._nearest_progress_index(state)



            # If close to the final expert index, directly finish when possible.

            if current_progress >= len(self.expert_path) - 2:

                if self.goal != state:

                    path.append(self.goal)

                break



            i = self.state_to_index[state]

            candidates = []



            for action_index, action in enumerate(ACTIONS):

                ns = self.transition(state, action)



                if ns == state:

                    continue



                if ns in visited and ns != self.goal:

                    continue



                next_progress = self._nearest_progress_index(ns)



                # Core fix: forbid obvious backward motion along the expert path.

                if next_progress < current_progress - 1:

                    continue



                progress_gain = next_progress - current_progress

                center_dist = nearest_centerline_distance(self.centerline_cells, ns)

                goal_dist = math.hypot(ns[0] - self.goal[0], ns[1] - self.goal[1])



                # Combine learned policy with explicit forward-progress preference.

                score = (

                    2.0 * policy[i, action_index]

                    + 0.20 * progress_gain

                    - 0.05 * center_dist

                    - 0.01 * goal_dist

                )



                candidates.append((score, ns))



            # If the strict no-revisit rule blocks all actions, relax it once,

            # but still prevent backward progress.

            if not candidates:

                for action_index, action in enumerate(ACTIONS):

                    ns = self.transition(state, action)

                    if ns == state:

                        continue



                    next_progress = self._nearest_progress_index(ns)

                    if next_progress < current_progress:

                        continue



                    center_dist = nearest_centerline_distance(self.centerline_cells, ns)

                    goal_dist = math.hypot(ns[0] - self.goal[0], ns[1] - self.goal[1])

                    score = (

                        2.0 * policy[i, action_index]

                        + 0.10 * (next_progress - current_progress)

                        - 0.05 * center_dist

                        - 0.01 * goal_dist

                    )

                    candidates.append((score, ns))



            if not candidates:

                print("Warning: no valid forward action found, stop early.")

                break



            candidates.sort(key=lambda x: x[0], reverse=True)

            state = candidates[0][1]



            path.append(state)

            visited.add(state)



            if state == self.goal:

                break



        return path



def evaluate_path(generated, expert):

    goal = expert[-1]



    # 平均到专家轨迹最近距离

    expert_arr = np.asarray(expert, dtype=np.float64)

    nearest_dists = []

    for p in generated:

        d = np.sqrt((expert_arr[:, 0] - p[0]) ** 2 + (expert_arr[:, 1] - p[1]) ** 2)

        nearest_dists.append(float(np.min(d)))



    fde = math.hypot(generated[-1][0] - goal[0], generated[-1][1] - goal[1])

    success = 1 if generated[-1] == goal else 0



    turns = 0

    prev = None

    for i in range(1, len(generated)):

        cur = (generated[i][0] - generated[i - 1][0],

               generated[i][1] - generated[i - 1][1])

        if prev is not None and cur != prev:

            turns += 1

        prev = cur



    return {

        "mean_distance_to_expert": float(np.mean(nearest_dists)),

        "FDE_to_expert_goal": float(fde),

        "generated_path_length_cells": len(generated),

        "turns": turns,

        "success": success,

    }





def plot_results(grid, expert_path, generated_path, losses, cfg):

    cfg.out_dir.mkdir(parents=True, exist_ok=True)



    plt.figure(figsize=(10, 8))

    plt.imshow(grid, origin="lower", alpha=0.75)



    ex = np.asarray(expert_path)

    gp = np.asarray(generated_path)



    plt.plot(ex[:, 0], ex[:, 1], linewidth=2.0, label="real nuPlan expert trajectory")

    plt.plot(gp[:, 0], gp[:, 1], linewidth=2.5, marker="o", markersize=2, label="IRL generated path")



    plt.scatter([expert_path[0][0]], [expert_path[0][1]], marker="s", s=80, label="start")

    plt.scatter([expert_path[-1][0]], [expert_path[-1][1]], marker="*", s=150, label="goal")



    plt.title("Real nuPlan expert trajectory and MaxEnt IRL path")

    plt.xlabel("grid x")

    plt.ylabel("grid y")

    plt.legend()

    plt.tight_layout()

    plt.savefig(cfg.out_dir / "real_irl_path.png", dpi=180)

    plt.close()



    plt.figure(figsize=(8, 4))

    plt.plot(losses)

    plt.title("Real nuPlan MaxEnt IRL training curve")

    plt.xlabel("epoch")

    plt.ylabel("feature expectation gap")

    plt.tight_layout()

    plt.savefig(cfg.out_dir / "real_irl_loss.png", dpi=180)

    plt.close()





def main():

    cfg = Config()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)



    print("Loading real nuPlan ego trajectory...")

    expert_path, raw_xy = load_expert_path(cfg)



    print(f"Expert path cells: {len(expert_path)}")

    print(f"Start cell: {expert_path[0]}")

    print(f"Goal cell: {expert_path[-1]}")



    print("Building local drivable grid from expert trajectory...")

    grid = build_grid_from_expert_path(expert_path, cfg)

    print(f"Grid shape: {grid.shape}, valid states: {np.sum(grid > 0)}")



    print("Training MaxEnt IRL on real nuPlan expert trajectory...")

    irl = MaxEntIRL(grid, expert_path, cfg)

    losses = irl.train()



    print("Generating path using learned reward...")

    generated = irl.generate_path()



    print("\nLearned reward weights:")

    names = ["goal_progress", "centerline_distance", "on_centerline", "goal_reaching"]

    for name, value in zip(names, irl.theta):

        print(f"{name:24s}: {value:.4f}")



    metrics = evaluate_path(generated, expert_path)



    print("\nEvaluation metrics:")

    for k, v in metrics.items():

        print(f"{k:28s}: {v}")



    plot_results(grid, expert_path, generated, losses, cfg)



    print("\nSaved:")

    print(cfg.out_dir / "real_irl_path.png")

    print(cfg.out_dir / "real_irl_loss.png")





if __name__ == "__main__":

    main()



#!/usr/bin/env python3

# -*- coding: utf-8 -*-



import math

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt





class Config:

    width = 80

    height = 30

    start = (5, 15)

    goal = (74, 15)

    gamma = 0.95

    beta = 5.0

    lr = 0.15

    epochs = 180

    horizon = 90

    out_dir = Path("outputs")





ACTIONS = [

    (1, 0),

    (1, -1),

    (1, 1),

    (0, -1),

    (0, 1),

    (0, 0),

]





def build_map(cfg):

    grid = np.zeros((cfg.height, cfg.width), dtype=np.int32)



    lane_ys = [10, 15, 20]

    for y in lane_ys:

        grid[y - 2:y + 3, :] = 1

        grid[y, :] = 2



    obstacles = [

        (25, 15), (26, 15), (27, 15),

        (45, 14), (45, 15), (45, 16),

        (58, 20), (59, 20), (60, 20),

    ]



    for x, y in obstacles:

        grid[y - 1:y + 2, x - 1:x + 2] = 3



    gx, gy = cfg.goal

    grid[gy, gx] = 4

    return grid





def valid_state(grid, state):

    x, y = state

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





def nearest_obstacle_distance(grid, state):

    ys, xs = np.where(grid == 3)

    if len(xs) == 0:

        return 999.0

    x, y = state

    dists = np.sqrt((xs - x) ** 2 + (ys - y) ** 2)

    return float(np.min(dists))





def nearest_centerline_distance(grid, state):

    ys, xs = np.where(grid == 2)

    x, y = state

    dists = np.sqrt((xs - x) ** 2 + (ys - y) ** 2)

    return float(np.min(dists))





def features_for_state(grid, cfg, state):

    x, y = state

    gx, gy = cfg.goal



    goal_dist = math.hypot(gx - x, gy - y)

    max_goal_dist = math.hypot(cfg.width, cfg.height)



    obs_dist = nearest_obstacle_distance(grid, state)

    center_dist = nearest_centerline_distance(grid, state)



    is_centerline = 1.0 if grid[y, x] == 2 else 0.0

    is_goal = 1.0 if state == cfg.goal else 0.0



    return np.array([

        -goal_dist / max_goal_dist,

        min(obs_dist, 10.0) / 10.0,

        -center_dist / 8.0,

        is_centerline,

        is_goal,

    ], dtype=np.float64)





def build_expert_trajectories(cfg):

    trajectories = []



    traj1 = []

    for x in range(cfg.start[0], cfg.goal[0] + 1):

        y = 15

        if 20 <= x <= 32:

            y = 10

        elif 40 <= x <= 50:

            y = 20

        traj1.append((x, y))

    trajectories.append(traj1)



    traj2 = []

    for x in range(cfg.start[0], cfg.goal[0] + 1):

        y = 15

        if 22 <= x <= 34:

            y = 10

        elif 39 <= x <= 51:

            y = 20

        traj2.append((x, y))

    trajectories.append(traj2)



    traj3 = []

    for x in range(cfg.start[0], cfg.goal[0] + 1):

        y = 15

        if 18 <= x <= 30:

            y = 10

        elif 42 <= x <= 52:

            y = 20

        traj3.append((x, y))

    trajectories.append(traj3)



    return trajectories





class MaxEntIRL:

    def __init__(self, grid, cfg):

        self.grid = grid

        self.cfg = cfg

        self.states = get_states(grid)

        self.state_to_index = {s: i for i, s in enumerate(self.states)}

        self.features = np.vstack([

            features_for_state(grid, cfg, s) for s in self.states

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



        for _ in range(80):

            new_values = np.zeros_like(values)



            for i, state in enumerate(self.states):

                q_values = []

                for action in ACTIONS:

                    ns = self.transition(state, action)

                    j = self.state_to_index[ns]

                    q_values.append(rewards[j] + self.cfg.gamma * values[j])



                q_values = np.array(q_values)

                z = self.cfg.beta * q_values

                max_z = np.max(z)

                new_values[i] = (

                    max_z + np.log(np.sum(np.exp(z - max_z)))

                ) / self.cfg.beta



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



            q_values = np.array(q_values)

            z = self.cfg.beta * q_values

            z -= np.max(z)

            probs = np.exp(z)

            probs /= np.sum(probs)

            policy[i] = probs



        return policy



    def expert_feature_mean(self, expert_trajectories):

        mu = np.zeros(self.features.shape[1], dtype=np.float64)

        count = 0



        for trajectory in expert_trajectories:

            for state in trajectory:

                if state in self.state_to_index:

                    mu += self.features[self.state_to_index[state]]

                    count += 1



        return mu / max(count, 1)



    def expected_state_visitation_frequency(self, policy, starts):

        d = np.zeros((self.cfg.horizon, len(self.states)), dtype=np.float64)



        for start in starts:

            if start in self.state_to_index:

                d[0, self.state_to_index[start]] += 1.0 / len(starts)



        for t in range(self.cfg.horizon - 1):

            for i, state in enumerate(self.states):

                if d[t, i] <= 0:

                    continue



                for action_index, action in enumerate(ACTIONS):

                    ns = self.transition(state, action)

                    j = self.state_to_index[ns]

                    d[t + 1, j] += d[t, i] * policy[i, action_index]



        return d.sum(axis=0) / self.cfg.horizon



    def train(self, expert_trajectories):

        rng = np.random.default_rng(0)

        self.theta = rng.normal(0.0, 0.05, size=self.features.shape[1])



        expert_mu = self.expert_feature_mean(expert_trajectories)

        starts = [trajectory[0] for trajectory in expert_trajectories]



        losses = []



        for epoch in range(self.cfg.epochs):

            policy = self.soft_value_iteration()

            svf = self.expected_state_visitation_frequency(policy, starts)

            model_mu = svf @ self.features



            gradient = expert_mu - model_mu

            self.theta += self.cfg.lr * gradient



            loss = float(np.linalg.norm(gradient))

            losses.append(loss)



            if epoch % 20 == 0 or epoch == self.cfg.epochs - 1:

                print(

                    f"epoch={epoch:03d}, "

                    f"feature_gap={loss:.6f}, "

                    f"theta={np.round(self.theta, 3)}"

                )



        return losses



    def generate_path(self, start):

        policy = self.soft_value_iteration()



        path = [start]

        state = start

        visited = {state: 1}



        for _ in range(120):

            if state == self.cfg.goal:

                break



            i = self.state_to_index[state]

            ranked_actions = np.argsort(policy[i])[::-1]



            chosen_next_state = state



            for action_index in ranked_actions:

                ns = self.transition(state, ACTIONS[int(action_index)])

                if visited.get(ns, 0) < 3:

                    chosen_next_state = ns

                    break



            state = chosen_next_state

            visited[state] = visited.get(state, 0) + 1

            path.append(state)



        return path





def evaluate_path(path, cfg):

    if len(path) <= 1:

        return {

            "path_length": 0.0,

            "turns": 0,

            "FDE_to_goal": 999.0,

            "success": 0,

        }



    path_length = 0.0

    turns = 0

    previous_direction = None



    for i in range(1, len(path)):

        dx = path[i][0] - path[i - 1][0]

        dy = path[i][1] - path[i - 1][1]



        path_length += math.hypot(dx, dy)



        current_direction = (dx, dy)

        if previous_direction is not None and current_direction != previous_direction:

            turns += 1



        previous_direction = current_direction



    fde_to_goal = math.hypot(

        path[-1][0] - cfg.goal[0],

        path[-1][1] - cfg.goal[1],

    )



    return {

        "path_length": path_length,

        "turns": turns,

        "FDE_to_goal": fde_to_goal,

        "success": 1 if path[-1] == cfg.goal else 0,

    }





def plot_results(grid, expert_trajectories, generated_path, losses, cfg):

    cfg.out_dir.mkdir(exist_ok=True)



    plt.figure(figsize=(12, 5))

    plt.imshow(grid, origin="lower")



    for index, trajectory in enumerate(expert_trajectories):

        xs = [p[0] for p in trajectory]

        ys = [p[1] for p in trajectory]

        label = "expert trajectories" if index == 0 else None

        plt.plot(xs, ys, linewidth=1.5, alpha=0.6, label=label)



    xs = [p[0] for p in generated_path]

    ys = [p[1] for p in generated_path]

    plt.plot(

        xs,

        ys,

        linewidth=2.5,

        marker="o",

        markersize=2,

        label="IRL generated path",

    )



    plt.scatter([cfg.start[0]], [cfg.start[1]], marker="s", s=80, label="start")

    plt.scatter([cfg.goal[0]], [cfg.goal[1]], marker="*", s=150, label="goal")



    plt.title("nuPlan-style map and MaxEnt IRL generated trajectory")

    plt.xlabel("x grid")

    plt.ylabel("y grid")

    plt.legend()

    plt.tight_layout()

    plt.savefig(cfg.out_dir / "quick_irl_path.png", dpi=180)

    plt.close()



    plt.figure(figsize=(8, 4))

    plt.plot(losses)

    plt.title("MaxEnt IRL training curve")

    plt.xlabel("epoch")

    plt.ylabel("feature expectation gap")

    plt.tight_layout()

    plt.savefig(cfg.out_dir / "quick_irl_loss.png", dpi=180)

    plt.close()





def main():

    cfg = Config()

    cfg.out_dir.mkdir(exist_ok=True)



    print("Building nuPlan-style semantic road map...")

    grid = build_map(cfg)



    print("Constructing expert trajectories...")

    expert_trajectories = build_expert_trajectories(cfg)



    print("Training MaxEnt IRL model...")

    irl = MaxEntIRL(grid, cfg)

    losses = irl.train(expert_trajectories)



    print("Generating path using learned reward function...")

    generated_path = irl.generate_path(cfg.start)



    print("\nLearned reward weights:")

    feature_names = [

        "goal_progress",

        "safe_distance",

        "centerline_distance",

        "on_centerline",

        "goal_reaching",

    ]



    for name, value in zip(feature_names, irl.theta):

        print(f"{name:24s}: {value:.4f}")



    metrics = evaluate_path(generated_path, cfg)



    print("\nEvaluation metrics:")

    for key, value in metrics.items():

        print(f"{key:24s}: {value}")



    print("\nGenerated path:")

    print(generated_path)



    plot_results(grid, expert_trajectories, generated_path, losses, cfg)



    print("\nSaved output files:")

    print(cfg.out_dir / "quick_irl_path.png")

    print(cfg.out_dir / "quick_irl_loss.png")





if __name__ == "__main__":

    main()


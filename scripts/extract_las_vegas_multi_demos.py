
#!/usr/bin/env python3

# -*- coding: utf-8 -*-



import sqlite3

from pathlib import Path

import json



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from tqdm import tqdm





DATA_ROOT = Path("data")

OUT_DIR = Path("outputs/las_vegas_multi_demo")

OUT_DIR.mkdir(parents=True, exist_ok=True)





def get_db_location(db_path):

    conn = sqlite3.connect(str(db_path))

    df = pd.read_sql_query("SELECT location, map_version, vehicle_name FROM log LIMIT 1;", conn)

    conn.close()

    if len(df) == 0:

        return None

    return df.loc[0, "location"]





def load_ego_xy(db_path):

    conn = sqlite3.connect(str(db_path))

    query = """

    SELECT

        lidar_pc.timestamp AS timestamp,

        ego_pose.x AS x,

        ego_pose.y AS y,

        ego_pose.vx AS vx,

        ego_pose.vy AS vy

    FROM lidar_pc

    JOIN ego_pose

    ON lidar_pc.ego_pose_token = ego_pose.token

    ORDER BY lidar_pc.timestamp ASC;

    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    return df





def trajectory_score(xy):

    if len(xy) < 20:

        return 0.0



    diff = np.diff(xy, axis=0)

    step = np.linalg.norm(diff, axis=1)

    path_len = np.sum(step)



    if path_len < 20:

        return 0.0



    net_dist = np.linalg.norm(xy[-1] - xy[0])

    straightness = net_dist / max(path_len, 1e-6)



    headings = np.unwrap(np.arctan2(diff[:, 1], diff[:, 0]))

    yaw_change = np.sum(np.abs(np.diff(headings)))



    return float(1.0 * yaw_change + 8.0 * (1.0 - straightness) + 0.005 * path_len)





def make_segments(df, db_path, window=900, step=300):

    records = []

    xy_all = df[["x", "y"]].to_numpy(dtype=float)



    for start in range(0, max(1, len(df) - window), step):

        seg = df.iloc[start:start + window].copy()

        if len(seg) < window // 2:

            continue



        xy = seg[["x", "y"]].to_numpy(dtype=float)

        score = trajectory_score(xy)



        if score <= 0:

            continue



        records.append({

            "db_path": str(db_path),

            "start": start,

            "end": start + len(seg),

            "score": score,

            "center_x": float(np.mean(xy[:, 0])),

            "center_y": float(np.mean(xy[:, 1])),

            "min_x": float(np.min(xy[:, 0])),

            "max_x": float(np.max(xy[:, 0])),

            "min_y": float(np.min(xy[:, 1])),

            "max_y": float(np.max(xy[:, 1])),

        })



    return records





def main():

    db_files = sorted(DATA_ROOT.rglob("*.db"))

    las_dbs = []



    for db in db_files:

        try:

            if get_db_location(db) == "las_vegas":

                las_dbs.append(db)

        except Exception:

            pass



    print(f"las_vegas db count: {len(las_dbs)}")



    all_segments = []



    for db in tqdm(las_dbs, desc="scan db"):

        try:

            df = load_ego_xy(db)

            segs = make_segments(df, db)

            all_segments.extend(segs)

        except Exception as e:

            print(f"skip {db}: {e}")



    seg_df = pd.DataFrame(all_segments)

    seg_df = seg_df.sort_values("score", ascending=False).reset_index(drop=True)

    seg_df.to_csv(OUT_DIR / "all_candidate_segments.csv", index=False)



    print("\nTop candidate segments:")

    print(seg_df.head(10).to_string(index=False))



    # 选择第一名作为地图中心，然后收集附近多条专家轨迹

    anchor = seg_df.iloc[0]

    anchor_x = anchor["center_x"]

    anchor_y = anchor["center_y"]



    selected = None



    for radius in [150, 250, 400, 600, 900]:

        d = np.sqrt((seg_df["center_x"] - anchor_x) ** 2 + (seg_df["center_y"] - anchor_y) ** 2)

        nearby = seg_df[d < radius].head(12).copy()



        if len(nearby) >= 4:

            selected = nearby

            print(f"\nSelected radius: {radius} m, demos: {len(selected)}")

            break



    if selected is None:

        selected = seg_df.head(8).copy()

        print("\nNot enough nearby segments, using top 8 globally.")



    selected.to_csv(OUT_DIR / "selected_segments.csv", index=False)



    # 提取 selected demos

    demo_rows = []

    meta = []



    for demo_id, row in selected.reset_index(drop=True).iterrows():

        db_path = Path(row["db_path"])

        df = load_ego_xy(db_path)

        seg = df.iloc[int(row["start"]):int(row["end"])].copy()



        # 下采样，减少状态点

        seg = seg.iloc[::8].copy()



        for t, (_, r) in enumerate(seg.iterrows()):

            demo_rows.append({

                "demo_id": int(demo_id),

                "t": int(t),

                "timestamp": int(r["timestamp"]),

                "x": float(r["x"]),

                "y": float(r["y"]),

                "vx": float(r["vx"]),

                "vy": float(r["vy"]),

                "db_path": str(db_path),

            })



        meta.append({

            "demo_id": int(demo_id),

            "db_path": str(db_path),

            "start": int(row["start"]),

            "end": int(row["end"]),

            "score": float(row["score"]),

        })



    demos = pd.DataFrame(demo_rows)

    demos.to_csv(OUT_DIR / "multi_expert_trajectories.csv", index=False)



    with open(OUT_DIR / "multi_expert_meta.json", "w", encoding="utf-8") as f:

        json.dump(meta, f, indent=2)



    # 可视化多条专家轨迹

    plt.figure(figsize=(10, 10))



    for demo_id, g in demos.groupby("demo_id"):

        plt.plot(g["x"], g["y"], linewidth=1.8, label=f"demo {demo_id}")

        plt.scatter([g["x"].iloc[0]], [g["y"].iloc[0]], marker="s", s=30)

        plt.scatter([g["x"].iloc[-1]], [g["y"].iloc[-1]], marker="*", s=60)



    plt.axis("equal")

    plt.xlabel("map x / m")

    plt.ylabel("map y / m")

    plt.title("Selected multi expert trajectories in Las Vegas")

    plt.legend()

    plt.tight_layout()

    plt.savefig(OUT_DIR / "multi_expert_trajectories.png", dpi=180)

    plt.close()



    print("\nSaved:")

    print(OUT_DIR / "all_candidate_segments.csv")

    print(OUT_DIR / "selected_segments.csv")

    print(OUT_DIR / "multi_expert_trajectories.csv")

    print(OUT_DIR / "multi_expert_trajectories.png")





if __name__ == "__main__":

    main()


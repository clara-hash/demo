
#!/usr/bin/env python3

# -*- coding: utf-8 -*-



import sqlite3

import sys

from pathlib import Path



import matplotlib.pyplot as plt

import pandas as pd





def get_columns(conn, table):

    cur = conn.cursor()

    cur.execute(f"PRAGMA table_info({table});")

    return [row[1] for row in cur.fetchall()]





def main():

    if len(sys.argv) < 2:

        print("Usage: python scripts/extract_ego_trajectory.py path/to/file.db")

        sys.exit(1)



    db_path = Path(sys.argv[1])

    out_dir = Path("outputs/nuplan_real")

    out_dir.mkdir(parents=True, exist_ok=True)



    conn = sqlite3.connect(str(db_path))



    tables = pd.read_sql_query(

        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;",

        conn,

    )["name"].tolist()



    print("Tables:", tables)



    if "ego_pose" not in tables:

        raise RuntimeError("No ego_pose table found in this db.")



    ego_cols = get_columns(conn, "ego_pose")

    print("ego_pose columns:", ego_cols)



    # nuPlan mini 通常可直接从 ego_pose 表中读取全局位姿。

    # 有些 db 需要通过 lidar_pc 的 timestamp 排序，这里优先使用 lidar_pc join ego_pose。

    if "lidar_pc" in tables:

        lidar_cols = get_columns(conn, "lidar_pc")

        print("lidar_pc columns:", lidar_cols)



        if "ego_pose_token" in lidar_cols and "token" in ego_cols and "timestamp" in lidar_cols:

            query = """

            SELECT

                lidar_pc.timestamp AS timestamp,

                ego_pose.x AS x,

                ego_pose.y AS y,

                ego_pose.z AS z,

                ego_pose.qw AS qw,

                ego_pose.qx AS qx,

                ego_pose.qy AS qy,

                ego_pose.qz AS qz

            FROM lidar_pc

            JOIN ego_pose

            ON lidar_pc.ego_pose_token = ego_pose.token

            ORDER BY lidar_pc.timestamp ASC;

            """

        else:

            query = """

            SELECT

                rowid AS timestamp,

                x, y, z, qw, qx, qy, qz

            FROM ego_pose

            ORDER BY rowid ASC;

            """

    else:

        query = """

        SELECT

            rowid AS timestamp,

            x, y, z, qw, qx, qy, qz

        FROM ego_pose

        ORDER BY rowid ASC;

        """



    df = pd.read_sql_query(query, conn)

    conn.close()



    print("\nExtracted ego trajectory:")

    print(df.head())

    print(df.tail())

    print(f"Total poses: {len(df)}")



    if len(df) == 0:

        raise RuntimeError("No ego trajectory extracted.")



    # 转成相对坐标，便于画图和后续 IRL 使用。

    df["rel_x"] = df["x"] - df["x"].iloc[0]

    df["rel_y"] = df["y"] - df["y"].iloc[0]



    csv_path = out_dir / "ego_trajectory.csv"

    df.to_csv(csv_path, index=False)



    plt.figure(figsize=(8, 8))

    plt.plot(df["rel_x"], df["rel_y"], linewidth=2)

    plt.scatter([df["rel_x"].iloc[0]], [df["rel_y"].iloc[0]], marker="s", s=80, label="start")

    plt.scatter([df["rel_x"].iloc[-1]], [df["rel_y"].iloc[-1]], marker="*", s=140, label="end")

    plt.axis("equal")

    plt.xlabel("relative x / m")

    plt.ylabel("relative y / m")

    plt.title("nuPlan ego expert trajectory")

    plt.legend()

    plt.tight_layout()



    fig_path = out_dir / "ego_trajectory.png"

    plt.savefig(fig_path, dpi=180)

    plt.close()



    print("\nSaved:")

    print(csv_path)

    print(fig_path)





if __name__ == "__main__":

    main()


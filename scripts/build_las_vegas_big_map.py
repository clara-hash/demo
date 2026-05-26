
#!/usr/bin/env python3

# -*- coding: utf-8 -*-



import sqlite3

from pathlib import Path



import geopandas as gpd

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import rasterio

from rasterio.features import rasterize

from rasterio.transform import from_origin

from shapely.geometry import box





GPKG_PATH = Path("data/maps/us-nv-las-vegas-strip/9.15.1915/map.gpkg")

DEMO_CSV = Path("outputs/las_vegas_multi_demo/multi_expert_trajectories.csv")

OUT_DIR = Path("outputs/las_vegas_big_map")



RESOLUTION = 2.0   # meters per cell. 1.0更细但会慢；2.0适合Jetson先跑通

MARGIN = 120.0     # meters





def query_epsg_from_db(db_path):

    try:

        conn = sqlite3.connect(str(db_path))

        df = pd.read_sql_query("SELECT epsg FROM ego_pose LIMIT 1;", conn)

        conn.close()

        if len(df) > 0 and not pd.isna(df.loc[0, "epsg"]):

            return int(df.loc[0, "epsg"])

    except Exception:

        pass

    return 32611  # Las Vegas UTM Zone 11N fallback





def read_layer(layer_name, epsg, bbox_poly):

    print(f"Reading layer: {layer_name}")

    gdf = gpd.read_file(GPKG_PATH, layer=layer_name)



    if gdf.crs is None:

        gdf = gdf.set_crs("EPSG:4326")



    gdf = gdf.to_crs(epsg=epsg)

    gdf = gdf[gdf.geometry.notnull()].copy()



    # Clip roughly to bounding box

    gdf = gdf[gdf.intersects(bbox_poly)].copy()



    if len(gdf) > 0:

        gdf["geometry"] = gdf.geometry.intersection(bbox_poly)

        gdf = gdf[~gdf.geometry.is_empty].copy()



    print(f"  kept rows: {len(gdf)}")

    return gdf





def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)



    if not DEMO_CSV.exists():

        raise FileNotFoundError(DEMO_CSV)



    demos = pd.read_csv(DEMO_CSV)

    first_db = demos["db_path"].iloc[0]

    epsg = query_epsg_from_db(first_db)

    print(f"Using EPSG:{epsg}")



    min_x = float(demos["x"].min() - MARGIN)

    max_x = float(demos["x"].max() + MARGIN)

    min_y = float(demos["y"].min() - MARGIN)

    max_y = float(demos["y"].max() + MARGIN)



    width = int(np.ceil((max_x - min_x) / RESOLUTION))

    height = int(np.ceil((max_y - min_y) / RESOLUTION))



    print(f"Map bounds:")

    print(f"  x: {min_x:.2f} ~ {max_x:.2f}")

    print(f"  y: {min_y:.2f} ~ {max_y:.2f}")

    print(f"Grid size: {width} x {height}, resolution={RESOLUTION} m")



    transform = from_origin(min_x, max_y, RESOLUTION, RESOLUTION)

    bbox_poly = box(min_x, min_y, max_x, max_y)



    # Semantic labels:

    # 0 non-drivable

    # 1 generic drivable

    # 2 lane

    # 3 lane connector

    # 4 intersection

    grid = np.zeros((height, width), dtype=np.uint8)



    layers = [

        ("generic_drivable_areas", 1),

        ("lanes_polygons", 2),

        ("gen_lane_connectors_scaled_width_polygons", 3),

        ("intersections", 4),

    ]



    for layer_name, value in layers:

        try:

            gdf = read_layer(layer_name, epsg, bbox_poly)

            if len(gdf) == 0:

                continue



            shapes = [(geom, value) for geom in gdf.geometry if geom is not None and not geom.is_empty]

            layer_grid = rasterize(

                shapes=shapes,

                out_shape=grid.shape,

                transform=transform,

                fill=0,

                all_touched=True,

                dtype=np.uint8,

            )



            mask = layer_grid > 0

            grid[mask] = layer_grid[mask]



        except Exception as e:

            print(f"Layer {layer_name} failed: {e}")



    # Convert demos to grid coordinates

    gx = np.floor((demos["x"].to_numpy() - min_x) / RESOLUTION).astype(int)

    gy = np.floor((max_y - demos["y"].to_numpy()) / RESOLUTION).astype(int)



    demos["gx"] = gx

    demos["gy"] = gy



    valid = (

        (demos["gx"] >= 0) & (demos["gx"] < width) &

        (demos["gy"] >= 0) & (demos["gy"] < height)

    )

    demos = demos[valid].copy()



    np.savez_compressed(

        OUT_DIR / "las_vegas_big_map.npz",

        grid=grid,

        min_x=min_x,

        max_y=max_y,

        resolution=RESOLUTION,

        epsg=epsg,

    )



    demos.to_csv(OUT_DIR / "multi_expert_trajectories_grid.csv", index=False)



    # Visualization

    plt.figure(figsize=(12, 10))

    plt.imshow(grid, origin="upper")



    for demo_id, g in demos.groupby("demo_id"):

        plt.plot(g["gx"], g["gy"], linewidth=1.5, label=f"demo {demo_id}")

        plt.scatter([g["gx"].iloc[0]], [g["gy"].iloc[0]], marker="s", s=30)

        plt.scatter([g["gx"].iloc[-1]], [g["gy"].iloc[-1]], marker="*", s=60)



    plt.title("Las Vegas big semantic map with multiple expert trajectories")

    plt.xlabel("grid x")

    plt.ylabel("grid y")

    plt.legend(loc="best", fontsize=8)

    plt.tight_layout()

    plt.savefig(OUT_DIR / "las_vegas_big_map_with_demos.png", dpi=180)

    plt.close()



    print("\nSaved:")

    print(OUT_DIR / "las_vegas_big_map.npz")

    print(OUT_DIR / "multi_expert_trajectories_grid.csv")

    print(OUT_DIR / "las_vegas_big_map_with_demos.png")





if __name__ == "__main__":

    main()


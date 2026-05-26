
#!/usr/bin/env python3

# -*- coding: utf-8 -*-



from pathlib import Path

import fiona

import geopandas as gpd





def main():

    gpkg_path = Path("data/maps/us-nv-las-vegas-strip/9.15.1915/map.gpkg")



    if not gpkg_path.exists():

        raise FileNotFoundError(gpkg_path)



    print(f"GPKG: {gpkg_path}")



    layers = fiona.listlayers(gpkg_path)

    print("\nLayers:")

    for layer in layers:

        print("  ", layer)



    print("\nLayer details:")

    for layer in layers:

        try:

            gdf = gpd.read_file(gpkg_path, layer=layer)

            print(f"\n[{layer}]")

            print(f"  rows: {len(gdf)}")

            print(f"  crs: {gdf.crs}")

            print(f"  columns: {list(gdf.columns)}")

            print(f"  geom types: {gdf.geometry.geom_type.value_counts().to_dict()}")

            print(f"  bounds: {gdf.total_bounds}")

        except Exception as e:

            print(f"\n[{layer}] read failed: {e}")





if __name__ == "__main__":

    main()


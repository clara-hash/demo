
#!/usr/bin/env python3

# -*- coding: utf-8 -*-



import sqlite3

from pathlib import Path

import pandas as pd



data_root = Path("data")

records = []



for db_path in sorted(data_root.rglob("*.db")):

    try:

        conn = sqlite3.connect(str(db_path))

        df = pd.read_sql_query("SELECT location, map_version, vehicle_name FROM log LIMIT 1;", conn)

        conn.close()



        if len(df) == 0:

            continue



        records.append({

            "db_path": str(db_path),

            "location": df.loc[0, "location"],

            "map_version": df.loc[0, "map_version"],

            "vehicle_name": df.loc[0, "vehicle_name"],

        })

    except Exception as e:

        records.append({

            "db_path": str(db_path),

            "location": f"ERROR: {e}",

            "map_version": "",

            "vehicle_name": "",

        })



out = pd.DataFrame(records)

out.to_csv("outputs/db_locations.csv", index=False)



print(out.groupby("location").size().sort_values(ascending=False))

print("\nSaved: outputs/db_locations.csv")



#!/usr/bin/env python3

# -*- coding: utf-8 -*-



import sqlite3

import sys

from pathlib import Path





def main():

    if len(sys.argv) < 2:

        print("Usage: python scripts/inspect_nuplan_db.py path/to/file.db")

        sys.exit(1)



    db_path = Path(sys.argv[1])

    if not db_path.exists():

        raise FileNotFoundError(db_path)



    print(f"DB file: {db_path}")

    print(f"Size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")



    conn = sqlite3.connect(str(db_path))

    cur = conn.cursor()



    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")

    tables = [row[0] for row in cur.fetchall()]



    print("\nTables:")

    for table in tables:

        cur.execute(f"SELECT COUNT(*) FROM {table};")

        count = cur.fetchone()[0]

        print(f"  {table:35s} rows={count}")



    print("\nColumns:")

    for table in tables:

        print(f"\n[{table}]")

        cur.execute(f"PRAGMA table_info({table});")

        for row in cur.fetchall():

            print(f"  {row[1]:30s} {row[2]}")



    conn.close()





if __name__ == "__main__":

    main()


# migrate_db.py
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "jev_eval.db")
con = sqlite3.connect(DB_PATH)
c = con.cursor()

# check existing columns
cols = [row[1] for row in c.execute("PRAGMA table_info(results)")]
print("existing columns:", cols)

if "noul_correct" not in cols:
    c.execute("ALTER TABLE results ADD COLUMN noul_correct INTEGER")
    con.commit()
    print("added noul_correct column")
else:
    print("noul_correct already exists")

# backfill for existing rows: noul_correct = (jev_prob >= 0.5) == ground_truth
c.execute("""
    UPDATE results
    SET noul_correct = CASE
        WHEN (jev_prob >= 0.5) = (ground_truth = 1) THEN 1
        ELSE 0
    END
    WHERE noul_correct IS NULL
""")
con.commit()
print("backfilled rows:", c.rowcount)

con.close()
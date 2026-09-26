import sqlite3
import numpy as np
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev_eval.db")


def compute_calibration(db_path=DB_PATH, n_bins=10):
    """Read results from SQLite and return calibration data.
    Returns an error dict if the table doesn't exist yet.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Guard: check the table exists before querying
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'")
    if not c.fetchone():
        conn.close()
        return {
            "bin_centers": [],
            "bin_accuracy": [],
            "bin_counts": [],
            "error": "No results table found. Run evaluate.py first."
        }

    c.execute("SELECT jev_prob, correct FROM results")
    rows = c.fetchall()
    conn.close()

    if not rows:
        return {
            "bin_centers": [],
            "bin_accuracy": [],
            "bin_counts": [],
            "error": "Results table is empty."
        }

    probs = [r[0] for r in rows]
    corrects = [r[1] for r in rows]

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_accuracy, bin_counts = [], [], []

    for i in range(len(bins) - 1):
        mask = [(p >= bins[i]) and (p < bins[i + 1]) for p in probs]
        count = sum(mask)
        if count > 0:
            acc = float(np.mean([x for x, m in zip(corrects, mask) if m]))
            bin_centers.append(float((bins[i] + bins[i + 1]) / 2))
            bin_accuracy.append(acc)
            bin_counts.append(count)

    return {
        "bin_centers": bin_centers,
        "bin_accuracy": bin_accuracy,
        "bin_counts": bin_counts,
        "error": None
    }

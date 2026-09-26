import sqlite3
import numpy as np
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev_eval.db")


def compute_confidence(db_path=DB_PATH, n_bins=10):
    """Read choice results and return confidence vs accuracy data."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Check if the results table and confidence column exist
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'")
    if not c.fetchone():
        conn.close()
        return {
            "bin_centers": [],
            "bin_accuracy": [],
            "bin_counts": [],
            "error": "No results table found. Run evaluate.py first."
        }

    c.execute("PRAGMA table_info(results)")
    columns = [row[1] for row in c.fetchall()]

    if "confidence" not in columns:
        conn.close()
        return {
            "bin_centers": [],
            "bin_accuracy": [],
            "bin_counts": [],
            "error": "No confidence column found in results table."
        }

    c.execute("SELECT confidence, correct FROM results WHERE confidence IS NOT NULL")
    rows = c.fetchall()
    conn.close()

    if not rows:
        return {
            "bin_centers": [],
            "bin_accuracy": [],
            "bin_counts": [],
            "error": "No confidence data available."
        }

    confidences = [r[0] for r in rows]
    corrects = [r[1] for r in rows]

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracy = []
    bin_counts = []

    for i in range(len(bins) - 1):
        mask = [(c >= bins[i]) and (c < bins[i + 1]) for c in confidences]
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

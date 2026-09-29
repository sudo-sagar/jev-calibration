'''
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
'''

import sqlite3
import os

import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev_eval.db")


def compute_calibration(db_path=DB_PATH, n_bins=10):
    """
    Read Noul probabilities and ground truth from SQLite.
    Bin predicted probabilities into deciles and compare against actual accuracy.

    Returns:
        {
          "bin_centers": [...],     # midpoint of each bin
          "bin_accuracy": [...],    # empirical accuracy in bin
          "bin_confidence": [...],  # mean predicted prob in bin
          "bin_counts": [...],
          "error": None | str
        }
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'")
    if not c.fetchone():
        conn.close()
        return {
            "bin_centers": [], "bin_accuracy": [], "bin_confidence": [],
            "bin_counts": [], "error": "No results table found. Run evaluate.py first."
        }

    c.execute(
        "SELECT jev_prob, ground_truth FROM results "
        "WHERE jev_prob IS NOT NULL AND ground_truth IS NOT NULL"
    )
    rows = c.fetchall()
    conn.close()

    if not rows:
        return {
            "bin_centers": [], "bin_accuracy": [], "bin_confidence": [],
            "bin_counts": [], "error": "Results table is empty."
        }

    probs = np.array([r[0] for r in rows], dtype=float)
    truths = np.array([r[1] for r in rows], dtype=float)

    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(probs, edges[1:-1], right=True)

    bin_centers, bin_accuracy, bin_confidence, bin_counts = [], [], [], []

    for i in range(n_bins):
        mask = bin_ids == i
        count = int(mask.sum())
        if count == 0:
            continue
        bin_centers.append(float((edges[i] + edges[i + 1]) / 2))
        bin_accuracy.append(float(truths[mask].mean()))
        bin_confidence.append(float(probs[mask].mean()))
        bin_counts.append(count)

    return {
        "bin_centers": bin_centers,
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "bin_counts": bin_counts,
        "error": None,
    }
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Binary ECE.
    y_true: (N,) 0/1 labels
    y_prob: (N,) predicted probability of the positive class
    Returns (ece, bin_data)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, edges[1:-1], right=True)

    ece = 0.0
    bin_acc, bin_conf, bin_count = [], [], []
    N = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        n_b = int(mask.sum())
        if n_b == 0:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            bin_count.append(0)
            continue
        acc_b = float(y_true[mask].mean())
        conf_b = float(y_prob[mask].mean())
        ece += (n_b / N) * abs(acc_b - conf_b)
        bin_acc.append(acc_b)
        bin_conf.append(conf_b)
        bin_count.append(n_b)

    return float(ece), {
        "bin_edges": edges.tolist(),
        "bin_acc": [None if np.isnan(x) else float(x) for x in bin_acc],
        "bin_conf": [None if np.isnan(x) else float(x) for x in bin_conf],
        "bin_count": bin_count,
    }
def multiclass_ece(y_true, prob_matrix, n_bins=10):
    """
    Standard multiclass ECE (confidence-based).

    y_true: (N,) integer class indices
    prob_matrix: (N, C) predicted probabilities (rows sum to 1)
    Returns (ece, bin_data)

    Bins predictions by max-prob confidence, compares mean confidence against
    empirical accuracy (fraction where argmax == y_true) in each bin.
    """
    y_true = np.asarray(y_true)
    prob_matrix = np.asarray(prob_matrix, dtype=float)

    conf = prob_matrix.max(axis=1)
    pred = prob_matrix.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    return expected_calibration_error(correct, conf, n_bins=n_bins)


def brier_multiclass(y_true, prob_matrix):
    """Mean squared error summed over classes (multi-class Brier)."""
    y_true = np.asarray(y_true)
    prob_matrix = np.asarray(prob_matrix, dtype=float)
    N, C = prob_matrix.shape

    y_onehot = np.zeros_like(prob_matrix)
    y_onehot[np.arange(N), y_true] = 1.0

    return float(np.mean(np.sum((prob_matrix - y_onehot) ** 2, axis=1)))
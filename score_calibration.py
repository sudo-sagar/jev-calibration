# score_calibration.py

import json
import os
import sqlite3

import numpy as np
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev_eval.db")


def load_score_data(db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT id, jev_prob, score_value, score_conf, score_probs, ground_truth "
        "FROM results WHERE score_probs IS NOT NULL AND jev_prob IS NOT NULL",
        con,
    )
    con.close()

    prob_matrix = []
    scores = []
    confs = []
    noul = []
    truths = []
    for _, row in df.iterrows():
        probs = json.loads(row["score_probs"])
        vec = [float(probs.get(str(k), probs.get(k, 0.0))) for k in range(5)]
        prob_matrix.append(vec)
        scores.append(float(row["score_value"]))
        confs.append(float(row["score_conf"]))
        noul.append(float(row["jev_prob"]))
        truths.append(int(row["ground_truth"]))

    return {
        "prob_matrix": np.array(prob_matrix),
        "score": np.array(scores),
        "confidence": np.array(confs),
        "noul": np.array(noul),
        "ground_truth": np.array(truths),
        "n": len(df),
    }


def mean_max_probability(prob_matrix):
    """Mean of the max probability per row — a simple concentration measure."""
    return float(prob_matrix.max(axis=1).mean())


def normalized_entropy(prob_matrix):
    """
    Shannon entropy normalized to [0, 1] by log(C).
    Lower = more concentrated (peakier).
    """
    C = prob_matrix.shape[1]
    eps = 1e-12
    H = -np.sum(prob_matrix * np.log(prob_matrix + eps), axis=1)
    return float((H / np.log(C)).mean())


def cumulative_monotonicity_score(prob_matrix):
    """
    Fraction of rows where P(y >= k) is non-increasing in k.
    Perfect monotonicity = 1.0.
    """
    cum = np.cumsum(prob_matrix[:, ::-1], axis=1)[:, ::-1]  # P(y >= k)
    diffs = np.diff(cum, axis=1)
    # non-increasing means diff <= small tolerance
    ok = (diffs <= 1e-6).all(axis=1)
    return float(ok.mean())


def spearman_with_noul(score, noul):
    from scipy.stats import spearmanr
    rho, _ = spearmanr(score, noul)
    return float(rho)


def analyze(db_path=DB_PATH):
    data = load_score_data(db_path)
    P = data["prob_matrix"]

    return {
        "n": data["n"],
        "mean_score_value": float(data["score"].mean()),
        "std_score_value": float(data["score"].std()),
        "mean_confidence": float(data["confidence"].mean()),
        "mean_max_probability": mean_max_probability(P),
        "normalized_entropy": normalized_entropy(P),
        "monotonicity_fraction": cumulative_monotonicity_score(P),
        "spearman_score_vs_noul": spearman_with_noul(data["score"], data["noul"]),
        "baseline_max_probability": 1.0 / P.shape[1],
    }


if __name__ == "__main__":
    import json as _j
    print(_j.dumps(analyze(), indent=2))
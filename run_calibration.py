# run_calibration.py
"""
Standalone calibration driver.

Reads Noul probabilities (jev_prob) and ground truth from jev_eval.db,
computes ECE + Brier before and after:
  (a) temperature scaling (single scalar, global)
  (b) isotonic regression (non-parametric, monotone)

Uses a held-out 50/50 split so calibrators are NOT fit on the eval data.
Writes calibration_metrics.json for the Streamlit dashboard.
"""
import json
import os
import sqlite3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from score_calibration import analyze as analyze_score
from sklearn.isotonic import IsotonicRegression
from calibration import multiclass_ece, brier_multiclass

from temperature import fit_temperature_binary, apply_temperature_binary

DB_PATH = os.path.join(os.path.dirname(__file__), "jev_eval.db")
OUT_PATH = os.path.join(os.path.dirname(__file__), "calibration_metrics.json")


# ---------------- metrics ----------------

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Binary ECE. Returns (ece, bin_data)."""
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


def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


# ---------------- data ----------------

def load_noul_data(db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT id, jev_prob, ground_truth FROM results "
        "WHERE jev_prob IS NOT NULL AND ground_truth IS NOT NULL",
        con,
    )
    con.close()

    assert df["jev_prob"].between(0, 1).all(), "jev_prob not in [0,1]"
    assert set(df["ground_truth"].unique()).issubset({0, 1}), \
        f"unexpected labels: {df['ground_truth'].unique()}"

    return df
def load_choice_data(db_path=DB_PATH):
    """Load choice label, full probability vector, and ground truth."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT id, choice, choice_probs, ground_truth FROM results "
        "WHERE choice_probs IS NOT NULL AND ground_truth IS NOT NULL",
        con,
    )
    con.close()

    prob_matrix = []
    y_true = []
    for _, row in df.iterrows():
        probs = json.loads(row["choice_probs"])   # {"yes": p, "no": p}
        p_yes = float(probs.get("yes", 0.0))
        p_no = float(probs.get("no", 1.0 - p_yes))
        prob_matrix.append([p_no, p_yes])         # class 0 = no, class 1 = yes
        y_true.append(int(row["ground_truth"]))   # 0 = no, 1 = yes

    return np.array(prob_matrix), np.array(y_true)

# ---------------- main ----------------

def main():
    df = load_noul_data()
    print(f"loaded {len(df)} rows")
    print(f"positive rate:  {df['ground_truth'].mean():.3f}")
    print(f"mean jev_prob:  {df['jev_prob'].mean():.3f}")

    y = df["ground_truth"].to_numpy()
    p = df["jev_prob"].to_numpy()

    idx = np.arange(len(df))
    idx_fit, idx_eval = train_test_split(
        idx, test_size=0.5, random_state=42, stratify=y
    )

    # ---------- before ----------
    ece_before, bins_before = expected_calibration_error(y[idx_eval], p[idx_eval])
    brier_before = brier_score(y[idx_eval], p[idx_eval])
    acc_before = float(((p[idx_eval] >= 0.5).astype(int) == y[idx_eval]).mean())

    # ---------- temperature scaling ----------
    T = fit_temperature_binary(y[idx_fit], p[idx_fit])
    p_temp = apply_temperature_binary(p[idx_eval], T)

    ece_temp, bins_temp = expected_calibration_error(y[idx_eval], p_temp)
    brier_temp = brier_score(y[idx_eval], p_temp)
    acc_temp = float(((p_temp >= 0.5).astype(int) == y[idx_eval]).mean())

    # ---------- isotonic regression ----------
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p[idx_fit], y[idx_fit])
    p_iso = iso.predict(p[idx_eval])

    ece_iso, bins_iso = expected_calibration_error(y[idx_eval], p_iso)
    brier_iso = brier_score(y[idx_eval], p_iso)
    acc_iso = float(((p_iso >= 0.5).astype(int) == y[idx_eval]).mean())

        # ---------- Choice (multiclass) ----------
    prob_matrix, y_choice = load_choice_data()
    print()
    print("=" * 60)
    print("Choice head (multiclass)")
    print("=" * 60)
    print(f"loaded {len(y_choice)} choice rows")
    print(f"class balance (no/yes): {(y_choice == 0).sum()}/{(y_choice == 1).sum()}")

    ece_choice, bins_choice = multiclass_ece(y_choice, prob_matrix, n_bins=10)
    brier_choice = brier_multiclass(y_choice, prob_matrix)
    acc_choice = float((prob_matrix.argmax(axis=1) == y_choice).mean())
    mean_conf_choice = float(prob_matrix.max(axis=1).mean())

    print(f"multiclass ECE         = {ece_choice:.4f}")
    print(f"multiclass Brier       = {brier_choice:.4f}")
    print(f"accuracy               = {acc_choice:.4f}")
    print(f"mean top-class conf    = {mean_conf_choice:.4f}")
        # ---------- Score (ordinal) ----------
    score_metrics = analyze_score()
    print()
    print("=" * 60)
    print("Score head (ordinal)")
    print("=" * 60)
    for k, v in score_metrics.items():
        if isinstance(v, float):
            print(f"  {k:32s} = {v:.4f}")
        else:
            print(f"  {k:32s} = {v}")

    # ---------- report ----------
    print()
    print("=" * 60)
    print(f"Temperature T                 = {T:.4f}")
    print(f"ECE   before                  = {ece_before:.4f}")
    print(f"ECE   after (temperature)     = {ece_temp:.4f}   "
          f"(delta {ece_temp - ece_before:+.4f})")
    print(f"ECE   after (isotonic)        = {ece_iso:.4f}   "
          f"(delta {ece_iso - ece_before:+.4f})")
    print(f"Brier before                  = {brier_before:.4f}")
    print(f"Brier after (temperature)     = {brier_temp:.4f}")
    print(f"Brier after (isotonic)        = {brier_iso:.4f}")
    print(f"Accuracy before               = {acc_before:.4f}")
    print(f"Accuracy after (temperature)  = {acc_temp:.4f}")
    print(f"Accuracy after (isotonic)     = {acc_iso:.4f}")
    print("=" * 60)

    metrics = {
        "n_rows": int(len(df)),
        "n_eval": int(len(idx_eval)),
        "positive_rate": float(df["ground_truth"].mean()),
        "mean_prob": float(df["jev_prob"].mean()),

        "temperature": T,
        "ece_before": ece_before,
        "ece_after_temperature": ece_temp,
        "ece_after_isotonic": ece_iso,
        "brier_before": brier_before,
        "brier_after_temperature": brier_temp,
        "brier_after_isotonic": brier_iso,
        "accuracy_before": acc_before,
        "accuracy_after_temperature": acc_temp,
        "accuracy_after_isotonic": acc_iso,
        "bins_before": bins_before,
        "bins_after_temperature": bins_temp,
        "bins_after_isotonic": bins_iso,

        # Choice head
        "choice_ece": ece_choice,
        "choice_brier": brier_choice,
        "choice_accuracy": acc_choice,
        "choice_mean_confidence": mean_conf_choice,
        "choice_bins": bins_choice,
        "score": score_metrics
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
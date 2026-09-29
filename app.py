import sqlite3
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# Data

conn = sqlite3.connect('jev_eval.db')
df = pd.read_sql("SELECT * FROM results", conn)
conn.close()

st.title("Jev Calibration Report")
st.markdown("Independent evaluation of Jev's calibrated probabilities.")


total = len(df)
accuracy = df['correct'].mean()

col1, col2 = st.columns(2)
col1.metric("Total Examples", total)
col2.metric("Accuracy", f"{accuracy:.1%}")

# ---------------------------------------------------------------------------
# Section 1: Simple calibration curve (hand-binned)
# ---------------------------------------------------------------------------
st.subheader("Calibration Curve")
bins = np.arange(0, 1.1, 0.1)
bin_centers = []
bin_accuracy = []

for i in range(len(bins) - 1):
    mask = (df['jev_prob'] >= bins[i]) & (df['jev_prob'] < bins[i + 1])
    if mask.sum() > 0:
        acc = df[mask]['correct'].mean()
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        bin_accuracy.append(acc)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(bin_centers, bin_accuracy, 'o-', label='Jev', color='blue')
ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Actual Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)


# Section 2: Reliability diagram — Before / Temperature / Isotonic

st.header("Probability Calibration — Reliability Diagram")

with open("calibration_metrics.json", encoding="utf-8") as f:
    m = json.load(f)


def _plot_curve(ax, bins_data, label, color):
    conf = np.array(
        [np.nan if x is None else x for x in bins_data["bin_conf"]], dtype=float
    )
    acc = np.array(
        [np.nan if x is None else x for x in bins_data["bin_acc"]], dtype=float
    )
    mask = ~np.isnan(conf) & ~np.isnan(acc)
    ax.plot(conf[mask], acc[mask], marker="o", label=label, color=color)


fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
_plot_curve(
    ax2, m["bins_before"],
    f"Before (ECE={m['ece_before']:.3f})",
    "tab:red",
)
_plot_curve(
    ax2, m["bins_after_temperature"],
    f"Temperature (ECE={m['ece_after_temperature']:.3f})",
    "tab:orange",
)
_plot_curve(
    ax2, m["bins_after_isotonic"],
    f"Isotonic (ECE={m['ece_after_isotonic']:.3f})",
    "tab:green",
)
ax2.set_xlabel("Predicted probability")
ax2.set_ylabel("Empirical accuracy")
ax2.set_title("Reliability diagram — Jev Noul on email dataset")
ax2.legend(loc="lower left")
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

st.subheader("Metrics")
st.table({
    "Metric": ["ECE", "Brier", "Accuracy"],
    "Before": [
        m["ece_before"],
        m["brier_before"],
        m["accuracy_before"],
    ],
    "Temperature": [
        m["ece_after_temperature"],
        m["brier_after_temperature"],
        m["accuracy_after_temperature"],
    ],
    "Isotonic": [
        m["ece_after_isotonic"],
        m["brier_after_isotonic"],
        m["accuracy_after_isotonic"],
    ],
})

st.caption(
    f"Temperature T = {m['temperature']:.3f} (T < 1 means model was under-confident). "
    f"n_eval = {m['n_eval']}. Positive rate = {m['positive_rate']:.3f}."
)
# ---------------------------------------------------------------------------
# Section: Choice head (multiclass) reliability
# ---------------------------------------------------------------------------
st.header("Choice Head — Multiclass Reliability")

choice_bins = m.get("choice_bins")
if choice_bins is None:
    st.info("No Choice calibration data. Run run_calibration.py after evaluate.py.")
else:
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    _plot_curve(
        ax3, choice_bins,
        f"Choice (ECE={m['choice_ece']:.3f})",
        "tab:purple",
    )
    ax3.set_xlabel("Top-class predicted probability (confidence)")
    ax3.set_ylabel("Empirical accuracy")
    ax3.set_title("Reliability diagram — Choice head (multiclass)")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

    st.table({
        "Metric": [
            "Multiclass ECE",
            "Multiclass Brier",
            "Accuracy",
            "Mean top-class confidence",
        ],
        "Value": [
            f"{m['choice_ece']:.4f}",
            f"{m['choice_brier']:.4f}",
            f"{m['choice_accuracy']:.4f}",
            f"{m['choice_mean_confidence']:.4f}",
        ],
    })

    st.caption(
        f"The Choice head is over-confident by "
        f"{m['choice_mean_confidence'] - m['choice_accuracy']:+.3f} "
        f"(mean top-class confidence − accuracy). By contrast, the Noul head is "
        f"under-confident by {m['mean_prob'] - m['positive_rate']:+.3f}."
    )

# Section 3: Raw results

st.subheader("Raw Results")
st.dataframe(df)
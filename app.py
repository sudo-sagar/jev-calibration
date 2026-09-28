import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

conn = sqlite3.connect('jev_eval.db')
df = pd.read_sql("SELECT * FROM results", conn)

st.title("Jev Calibration Report")
st.markdown("Independent evaluation of Jev's calibrated probabilities.")

# Summary stats
total = len(df)
accuracy = df['correct'].mean()
st.metric("Total Examples", total)
st.metric("Accuracy", f"{accuracy:.1%}")

# Calibration plot
st.subheader("Calibration Curve")
bins = np.arange(0, 1.1, 0.1)
bin_centers = []
bin_accuracy = []

for i in range(len(bins)-1):
    mask = (df['jev_prob'] >= bins[i]) & (df['jev_prob'] < bins[i+1])
    if mask.sum() > 0:
        acc = df[mask]['correct'].mean()
        bin_centers.append((bins[i] + bins[i+1]) / 2)
        bin_accuracy.append(acc)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(bin_centers, bin_accuracy, 'o-', label='Jev', color='blue')
ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Actual Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Raw data
st.subheader("Raw Results")
st.dataframe(df)

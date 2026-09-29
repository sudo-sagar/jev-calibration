# temperature.py
"""
Temperature scaling for probability calibration.

Fits a single scalar T on a held-out split to reduce miscalibration.
Does NOT change argmax predictions — only the confidence values.

For binary problems:
    logit = log(p / (1 - p))
    p_cal = sigmoid(logit / T)

T < 1  -> sharpens (fixes under-confidence)
T > 1  -> softens  (fixes over-confidence)
T = 1  -> no change
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax


def _probs_to_logits_binary(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def apply_temperature_binary(y_prob, T):
    """Rescale binary probabilities with temperature T."""
    y_prob = np.asarray(y_prob, dtype=float)
    logits = _probs_to_logits_binary(y_prob) / T
    return 1.0 / (1.0 + np.exp(-logits))


def apply_temperature_multiclass(y_prob_matrix, T):
    """Rescale multiclass probabilities with temperature T."""
    y_prob_matrix = np.asarray(y_prob_matrix, dtype=float)
    logits = np.log(np.clip(y_prob_matrix, 1e-12, 1.0)) / T
    return softmax(logits, axis=1)


def fit_temperature_binary(y_val, p_val):
    """
    Find scalar T minimizing NLL on validation set.
    y_val: (N,) 0/1 labels
    p_val: (N,) predicted probabilities in (0, 1)
    Returns: T (float)
    """
    y_val = np.asarray(y_val, dtype=float)
    p_val = np.asarray(p_val, dtype=float)

    def nll(T):
        p = apply_temperature_binary(p_val, T)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def fit_temperature_multiclass(y_val, prob_matrix_val):
    """
    Find scalar T minimizing NLL for multiclass.
    y_val: (N,) integer labels
    prob_matrix_val: (N, C) predicted probabilities
    Returns: T (float)
    """
    y_val = np.asarray(y_val)
    prob_matrix_val = np.asarray(prob_matrix_val, dtype=float)
    N = len(y_val)

    def nll(T):
        p = apply_temperature_multiclass(prob_matrix_val, T)
        return -np.mean(np.log(np.clip(p[np.arange(N), y_val], 1e-12, 1.0)))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)
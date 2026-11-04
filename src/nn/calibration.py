"""Calibration metrics and temperature scaling."""
import numpy as np
from scipy.optimize import minimize_scalar
from .mlp import softmax


def nll(probs, y):
    return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())


def ece(probs, y, bins=15):
    """Expected calibration error with equal-width confidence bins."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability(probs, y, bins=15):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def fit_temperature(logits, y):
    """Single scalar T minimising validation NLL of softmax(logits / T)."""
    f = lambda T: nll(softmax(logits / T), y)
    return float(minimize_scalar(f, bounds=(0.05, 10.0), method="bounded").x)


def confusion_matrix(pred, y, n=9):
    m = np.zeros((n, n), int)
    np.add.at(m, (y, pred), 1)
    return m

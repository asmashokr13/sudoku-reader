"""Train the MLP, fit temperature on validation only, compare with logistic regression.

python -m src.nn.train
Outputs: artifacts/mlp.npz, results/nn_results.json, results/nn_*.png
"""
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from config import ARTIFACT_DIR, RESULT_DIR, TEAM_SEED
from src.data.mnist import load_splits
from .mlp import MLP, softmax
from . import calibration as cal

EPOCHS = 10


def _metrics(name, logits, y, T):
    raw, ts = softmax(logits), softmax(logits / T)
    return {"accuracy": float((raw.argmax(1) == y).mean()),
            "nll_raw": cal.nll(raw, y), "nll_ts": cal.nll(ts, y),
            "ece_raw": cal.ece(raw, y), "ece_ts": cal.ece(ts, y), "T": T}


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True); RESULT_DIR.mkdir(exist_ok=True)
    d = load_splits()
    x = {k: d[f"x_{k}"].astype(np.float32).reshape(len(d[f"y_{k}"]), -1) / 255 for k in ("train", "val", "test")}
    y = {k: d[f"y_{k}"].astype(int) - 1 for k in ("train", "val", "test")}   # digit 1-9 -> class 0-8

    t0 = time.time()
    net = MLP(seed=TEAM_SEED)
    net.fit(x["train"], y["train"], epochs=EPOCHS, seed=TEAM_SEED, x_val=x["val"], y_val=y["val"])
    net_time = time.time() - t0
    T = cal.fit_temperature(net.logits(x["val"]), y["val"])      # validation only
    net.temperature = T
    net.save(ARTIFACT_DIR / "mlp.npz")

    t0 = time.time()
    lr = LogisticRegression(max_iter=100, random_state=TEAM_SEED)
    lr.fit(x["train"], y["train"])
    lr_time = time.time() - t0
    lr_val = lr.decision_function(x["val"]); lr_test = lr.decision_function(x["test"])
    lr_T = cal.fit_temperature(lr_val, y["val"])

    res = {"mlp": {"val": _metrics("mlp", net.logits(x["val"]), y["val"], T),
                   "test": _metrics("mlp", net.logits(x["test"]), y["test"], T),
                   "train_seconds": net_time, "sizes": net.sizes, "epochs": EPOCHS},
           "logreg": {"val": _metrics("lr", lr_val, y["val"], lr_T),
                      "test": _metrics("lr", lr_test, y["test"], lr_T), "train_seconds": lr_time}}
    cm_mlp = cal.confusion_matrix(net.logits(x["test"]).argmax(1), y["test"])
    cm_lr = cal.confusion_matrix(lr_test.argmax(1), y["test"])
    res["mlp"]["confusion_test"] = cm_mlp.tolist(); res["logreg"]["confusion_test"] = cm_lr.tolist()
    (RESULT_DIR / "nn_results.json").write_text(json.dumps(res, indent=2))

    # plots ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, cm, title in zip(ax, (cm_mlp, cm_lr), ("MLP", "Logistic regression")):
        a.imshow(cm, cmap="Blues"); a.set_title(f"{title}: confusion (test)")
        a.set_xticks(range(9)); a.set_xticklabels(range(1, 10)); a.set_yticks(range(9)); a.set_yticklabels(range(1, 10))
        a.set_xlabel("predicted"); a.set_ylabel("true")
        for i in range(9):
            for j in range(9):
                a.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(RESULT_DIR / "nn_confusion.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for a, logit, Tm, title in ((ax[0], net.logits(x["test"]), T, "MLP"), (ax[1], lr_test, lr_T, "Logistic regression")):
        for lab, P in (("raw", softmax(logit)), (f"temperature T={Tm:.2f}", softmax(logit / Tm))):
            xs, ys, _ = cal.reliability(P, y["test"]); a.plot(xs, ys, "o-", label=f"{lab} (ECE {cal.ece(P, y['test']):.4f})")
        a.plot([0, 1], [0, 1], "k--", lw=0.8); a.set_title(f"{title}: reliability (test)")
        a.set_xlabel("confidence"); a.legend(fontsize=8)
    ax[0].set_ylabel("accuracy")
    fig.tight_layout(); fig.savefig(RESULT_DIR / "nn_reliability.png", dpi=130); plt.close(fig)
    for k in ("mlp", "logreg"):
        print(k, {s: {m: round(v, 4) for m, v in res[k][s].items()} for s in ("val", "test")})


if __name__ == "__main__":
    main()

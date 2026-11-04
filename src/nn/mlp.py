"""Small MLP classifier in pure NumPy (784-128-64-9, ReLU, softmax, Adam)."""
import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, sizes=(784, 128, 64, 9), seed=0, temperature=1.0):
        rng = np.random.default_rng(seed)
        self.sizes = tuple(sizes)
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.temperature = float(temperature)

    # ---- forward -------------------------------------------------------
    def logits(self, x):
        h = x.reshape(len(x), -1).astype(np.float32)
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.maximum(h @ W + b, 0)
        return h @ self.W[-1] + self.b[-1]

    def predict_proba(self, x, calibrated=True):
        T = self.temperature if calibrated else 1.0
        return softmax(self.logits(x) / T)

    # ---- training ------------------------------------------------------
    def fit(self, x, y, epochs=10, batch=128, lr=1e-3, weight_decay=1e-4, seed=0,
            x_val=None, y_val=None, log=print):
        """y holds class indices 0..8."""
        rng = np.random.default_rng(seed)
        x = x.reshape(len(x), -1).astype(np.float32)
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        t, b1, b2, eps = 0, 0.9, 0.999, 1e-8
        hist = []
        for ep in range(epochs):
            order = rng.permutation(len(x))
            tot = 0.0
            for s in range(0, len(x), batch):
                idx = order[s:s + batch]
                xb, yb = x[idx], y[idx]
                acts = [xb]
                h = xb
                for W, b in zip(self.W[:-1], self.b[:-1]):
                    h = np.maximum(h @ W + b, 0)
                    acts.append(h)
                p = softmax(h @ self.W[-1] + self.b[-1])
                tot += -np.log(p[np.arange(len(yb)), yb] + 1e-12).sum()
                d = p; d[np.arange(len(yb)), yb] -= 1; d /= len(yb)
                t += 1
                for l in range(len(self.W) - 1, -1, -1):
                    gW = acts[l].T @ d + weight_decay * self.W[l]
                    gb = d.sum(axis=0)
                    if l > 0:
                        d = (d @ self.W[l].T) * (acts[l] > 0)
                    for P, g, m, v in ((self.W, gW, mW, vW), (self.b, gb, mb, vb)):
                        m[l] = b1 * m[l] + (1 - b1) * g
                        v[l] = b2 * v[l] + (1 - b2) * g * g
                        P[l] -= lr * (m[l] / (1 - b1 ** t)) / (np.sqrt(v[l] / (1 - b2 ** t)) + eps)
            msg = f"epoch {ep + 1}/{epochs} train-NLL {tot / len(x):.4f}"
            if x_val is not None:
                acc = (self.logits(x_val).argmax(1) == y_val).mean()
                msg += f"  val-acc {acc:.4f}"
            hist.append(tot / len(x))
            log(msg)
        return hist

    # ---- persistence ---------------------------------------------------
    def save(self, path):
        np.savez(path, sizes=np.array(self.sizes), temperature=self.temperature,
                 **{f"W{i}": w for i, w in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls(tuple(int(s) for s in z["sizes"]), temperature=float(z["temperature"]))
        m.W = [z[f"W{i}"] for i in range(len(m.W))]
        m.b = [z[f"b{i}"] for i in range(len(m.b))]
        return m

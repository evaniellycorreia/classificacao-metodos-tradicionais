"""Classificadores lineares em torch: regressão logística e SVM linear (hinge) na forma primal."""
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin


class TorchLinearClassifier(ClassifierMixin, BaseEstimator):
    """Minimiza mean(cw_i * perda_i) + ||w||^2 / (2 C n), equivalente ao objetivo de C * soma(cw_i * perda_i) + ||w||^2 / 2
    usado por sklearn (SVC linear e LogisticRegression) com class_weight="balanced"."""

    def __init__(self, loss="logistic", C=1.0, epochs=500, lr=0.05, random_state=0):
        self.loss = loss
        self.C = C
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        counts = np.bincount(y, minlength=2)
        class_w = len(y) / (2.0 * counts)
        sample_w = torch.as_tensor(class_w[y], dtype=torch.float32)
        sign = torch.as_tensor(2.0 * y - 1.0, dtype=torch.float32)
        n, d = X.shape

        w = torch.zeros(d, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        for _ in range(self.epochs):
            opt.zero_grad()
            margin = sign * (X @ w + b)
            if self.loss == "logistic":
                per_sample = torch.nn.functional.softplus(-margin)
            elif self.loss == "hinge":
                per_sample = torch.clamp(1.0 - margin, min=0.0)
            else:
                raise ValueError(f"loss desconhecida: {self.loss}")
            objective = (sample_w * per_sample).mean() + (w @ w) / (2.0 * self.C * n)
            objective.backward()
            opt.step()
            sched.step()

        self.coef_ = w.detach().numpy()[None, :]
        self.intercept_ = b.detach().numpy()
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=np.float32) @ self.coef_[0] + self.intercept_[0]

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

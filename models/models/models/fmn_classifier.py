from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class FMNConfig:
    max_hyperbox_size: float = 0.3
    gamma: float = 1.0
    n_epochs: int = 1
    shuffle: bool = True
    random_state: int = 42


class _Hyperbox:
    def __init__(self, V, W, label):
        self.V = V
        self.W = W
        self.label = label


class FuzzyMinMaxClassifier:
    def __init__(self, n_features, n_classes, config=None):
        self.n_features = n_features
        self.n_classes = n_classes
        self.config = config or FMNConfig()
        self.hyperboxes: List[_Hyperbox] = []
        self._rng = random.Random(self.config.random_state)

    def _membership(self, x, box):
        gamma = self.config.gamma
        n = self.n_features
        def f(r):
            return np.maximum(0.0, 1.0 - np.maximum(0.0, np.minimum(1.0, gamma * r)))
        return float(np.sum(f(x - box.W) + f(box.V - x)) / (2.0 * n))

    def _can_expand(self, x, box):
        new_V = np.minimum(box.V, x)
        new_W = np.maximum(box.W, x)
        return bool(np.all(new_W - new_V <= self.config.max_hyperbox_size))

    def _expand(self, x, box):
        box.V = np.minimum(box.V, x)
        box.W = np.maximum(box.W, x)

    def _overlap_exists(self, b1, b2):
        return bool(np.all(b1.V <= b2.W) and np.all(b2.V <= b1.W))

    def _contract(self, b1, b2):
        for i in range(self.n_features):
            if b1.V[i] < b2.V[i] and b1.W[i] > b2.V[i] and b1.W[i] < b2.W[i]:
                mid = (b1.W[i] + b2.V[i]) / 2.0
                b1.W[i] = mid
                b2.V[i] = mid
            elif b2.V[i] < b1.V[i] and b2.W[i] > b1.V[i] and b2.W[i] < b1.W[i]:
                mid = (b2.W[i] + b1.V[i]) / 2.0
                b2.W[i] = mid
                b1.V[i] = mid

    def fit(self, X, y):
        indices = list(range(len(X)))
        for epoch in range(self.config.n_epochs):
            if self.config.shuffle:
                self._rng.shuffle(indices)
            for idx in indices:
                x = X[idx]
                label = int(y[idx])
                best_box = None
                best_membership = -1.0
                for box in self.hyperboxes:
                    if box.label != label:
                        continue
                    if not self._can_expand(x, box):
                        continue
                    m = self._membership(x, box)
                    if m > best_membership:
                        best_membership = m
                        best_box = box
                if best_box is not None:
                    self._expand(x, best_box)
                    for other in self.hyperboxes:
                        if other is best_box or other.label == label:
                            continue
                        if self._overlap_exists(best_box, other):
                            self._contract(best_box, other)
                else:
                    self.hyperboxes.append(_Hyperbox(x.copy(), x.copy(), label))
        return self

    def predict_proba(self, X):
        proba = np.zeros((len(X), self.n_classes), dtype=np.float64)
        for i, x in enumerate(X):
            for box in self.hyperboxes:
                m = self._membership(x, box)
                if m > proba[i, box.label]:
                    proba[i, box.label] = m
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return proba / row_sums

    def predict(self, X):
        if not self.hyperboxes:
            return np.zeros(len(X), dtype=np.int64)
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

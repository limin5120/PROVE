"""L3 model heads: one root-cause head and one category head.

Why two heads and not five
--------------------------
The score has four components (AD 40, RCA 40, Major 10, Minor 10) but only two genuinely
distinct prediction problems once the feature frame is per (peak, candidate element):

* ``root`` -- binary "is this element the root cause of a real fault at this minute".  Taking
  the max over the ten roles of a peak gives that peak's fault probability (which drives
  selection and therefore the emission count, AD and alpha_fp), while the per-role values are
  already the Top-5 ranking the RCA term needs.  One head, two score components, and the two
  cannot disagree with each other by construction.
* ``cat`` -- 28-way (major, sub) over the official taxonomy, evaluated on the rank-1 role.

Window placement is deliberately NOT learned.  The labelled anchor can only teach the window
widths that batch 1 happened to have, and two measured attempts to impose a duration prior on
top of a learned selector both lost points (-3.04 and -3.36 on the stand-in).  The crossing
window on the element's own envelope is a physical estimate -- it measures how long the
elevation actually lasts -- so it transfers across batches without being fitted to one.

Only Segment 0 has labels, so both heads are trained there and applied to all three segments.
That is legitimate only because the evidence layer has already been aligned to Segment 0's
distribution (``evidence.align_segments_to_anchor``); without alignment the other segments are
out of distribution and the heads silently degrade, which is exactly how batch 3 lost 22.9
points in the previous architecture.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C  # noqa: E402

PARAMS_DUR = dict(objective="regression", metric="l2", learning_rate=0.05, num_leaves=31,
                  min_child_samples=25, feature_fraction=0.7, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=10.0, verbose=-1, n_jobs=-1, seed=7)
PARAMS_EVENT = dict(objective="binary", learning_rate=0.04, num_leaves=31, min_child_samples=60,
                    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                    lambda_l2=10.0, verbose=-1, n_jobs=-1, seed=7)
PARAMS_ROOT = dict(objective="binary", learning_rate=0.045, num_leaves=63, min_child_samples=40,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                   lambda_l2=5.0, verbose=-1, n_jobs=-1, seed=7)
PARAMS_CAT = dict(objective="multiclass", num_class=len(C.CATEGORY_PAIRS), learning_rate=0.05,
                  num_leaves=31, min_child_samples=20, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1,
                  n_jobs=-1, seed=7)
N_ROUNDS = 420
N_FOLDS = 5


class Head:
    """Thin LightGBM wrapper: fit / out-of-fold predict / save / load, one file per head."""

    def __init__(self, name: str, params: dict, cols: list[str]):
        self.name = name
        self.params = dict(params)
        self.cols = list(cols)
        self.models: list = []

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return df.reindex(columns=self.cols).fillna(0.0).to_numpy(dtype=np.float32)

    def fit_oof(self, df: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                n_folds: int = N_FOLDS, n_rounds: int = N_ROUNDS) -> np.ndarray:
        """Grouped K-fold: all ten roles of one peak must stay in the same fold, otherwise the
        model sees the same minute twice and the OOF numbers become fiction."""
        import lightgbm as lgb

        gid = pd.factorize(groups)[0]
        uniq = np.unique(gid)
        rng = np.random.default_rng(11)
        rng.shuffle(uniq)
        fold_of_group = {g: i % n_folds for i, g in enumerate(uniq)}
        fold = np.array([fold_of_group[g] for g in gid])
        X = self._X(df)
        w = self._weights(y)
        oof = np.zeros((len(y), self.params.get("num_class", 1) if "num_class" in self.params
                        else 1), dtype=np.float64)
        self.models = []
        for k in range(n_folds):
            tr, va = fold != k, fold == k
            if not tr.any() or not va.any():
                continue
            ds = lgb.Dataset(X[tr], label=y[tr], weight=w[tr] if w is not None else None)
            m = lgb.train(self.params, ds, num_boost_round=n_rounds,
                          valid_sets=[lgb.Dataset(X[va], label=y[va], reference=ds)],
                          callbacks=[lgb.early_stopping(60, verbose=False)])
            self.models.append(m)
            p = m.predict(X[va], num_iteration=m.best_iteration)
            oof[va] = p[:, None] if p.ndim == 1 else p
        return oof

    @staticmethod
    def _weights(y: np.ndarray):
        """Class balance: a matched root cause is ~1% of the (peak, role) frame."""
        pos = float((y == 1).sum()) if y.ndim == 1 else 0.0
        if pos <= 0:
            return None
        neg = float(len(y) - pos)
        w = np.ones(len(y), dtype=np.float64)
        w[y == 1] = min(neg / pos, 60.0)
        return w

    def predict(self, df: pd.DataFrame, models=None) -> np.ndarray:
        ms = models or self.models
        if not ms:
            raise SystemExit(f"head {self.name} has no fitted model")
        X = self._X(df)
        P = np.stack([m.predict(X, num_iteration=getattr(m, "best_iteration", None))
                      for m in ms])
        return P.mean(axis=0)

    def save(self, d: str) -> None:
        os.makedirs(d, exist_ok=True)
        for i, m in enumerate(self.models):
            m.save_model(os.path.join(d, f"{self.name}_{i}.txt"))
        with open(os.path.join(d, f"{self.name}_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "params": self.params, "cols": self.cols,
                       "n_models": len(self.models),
                       "best_iterations": [getattr(m, "best_iteration", None)
                                           for m in self.models]}, f, indent=1)

    @classmethod
    def load(cls, d: str, name: str) -> "Head":
        import lightgbm as lgb

        with open(os.path.join(d, f"{name}_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        h = cls(name, meta["params"], meta["cols"])
        h.models = [lgb.Booster(model_file=os.path.join(d, f"{name}_{i}.txt"))
                    for i in range(meta["n_models"])]
        return h

    def importance(self, top: int = 15) -> list[tuple[str, float]]:
        if not self.models:
            return []
        imp = np.mean([m.feature_importance("gain") for m in self.models], axis=0)
        order = np.argsort(-imp)[:top]
        tot = max(float(imp.sum()), 1e-9)
        return [(self.cols[i], round(float(imp[i]) / tot, 4)) for i in order]

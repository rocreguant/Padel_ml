"""Win-probability model: train, evaluate chronologically, persist, predict."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .db import Database
from .features import FEATURES, build_training_set, fixture_features

log = logging.getLogger(__name__)

MODEL_FILE = "winner_model.joblib"


def _augment(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add every match seen from the other side (features negated, label flipped).

    Removes any bias from how the source orders teams (e.g. winner listed first)."""
    return np.vstack([X, -X]), np.concatenate([y, 1 - y])


def _candidates() -> dict:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000,
                                                                     fit_intercept=False)),
        "gbm": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                              l2_regularization=1.0, min_samples_leaf=40),
    }


def symmetric_proba(model, X: np.ndarray) -> np.ndarray:
    """P(A wins) averaged with 1 - P(B wins) so p(A,B) + p(B,A) == 1."""
    return (model.predict_proba(X)[:, 1] + 1 - model.predict_proba(-X)[:, 1]) / 2


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "n": int(len(y)),
    }


@dataclass
class TrainReport:
    chosen: str
    validation: dict[str, dict[str, float]]
    n_train_matches: int


def _split(db: Database, holdout_frac: float, min_matches: int):
    matches = db.matches_df()
    ds, _ = build_training_set(matches, db.rankings_df())
    if len(ds) < min_matches:
        raise RuntimeError(f"only {len(ds)} completed matches - need at least {min_matches}. "
                           "Scrape/import more results first.")
    # Chronological split: train on the past, validate on the most recent matches.
    # Skip the first matches where every rating is still at its default.
    warm = min(len(ds) // 10, 200)
    ds_fit = ds.iloc[warm:]
    cut = int(len(ds_fit) * (1 - holdout_frac))
    return ds, ds_fit, ds_fit.iloc[:cut], ds_fit.iloc[cut:]


def holdout_predictions(db: Database, model_name: str = "logreg",
                        holdout_frac: float = 0.3, min_matches: int = 50) -> pd.DataFrame:
    """Out-of-sample P(A wins) for the most recent `holdout_frac` of matches, from a
    model trained only on earlier matches. Used for honest betting backtests."""
    _, _, tr, va = _split(db, holdout_frac, min_matches)
    mdl = _candidates()[model_name]
    mdl.fit(*_augment(tr[FEATURES].to_numpy(float), tr["y"].to_numpy(int)))
    out = va[["match_id", "date", "y", "elo_prob"]].copy()
    out["prob_a"] = symmetric_proba(mdl, va[FEATURES].to_numpy(float))
    return out


def train(db: Database, model_dir: Path, holdout_frac: float = 0.2,
          min_matches: int = 50) -> TrainReport:
    ds, ds_fit, tr, va = _split(db, holdout_frac, min_matches)
    Xtr, ytr = _augment(tr[FEATURES].to_numpy(float), tr["y"].to_numpy(int))
    Xva, yva = va[FEATURES].to_numpy(float), va["y"].to_numpy(int)

    results = {"elo_baseline": _metrics(yva, va["elo_prob"].to_numpy())}
    fitted = {}
    for name, mdl in _candidates().items():
        mdl.fit(Xtr, ytr)
        fitted[name] = mdl
        results[name] = _metrics(yva, symmetric_proba(mdl, Xva))
    for k, v in results.items():
        log.info("%-13s logloss=%.4f brier=%.4f acc=%.3f", k, v["log_loss"], v["brier"],
                 v["accuracy"])

    chosen = min(fitted, key=lambda k: results[k]["log_loss"])
    # refit chosen model on everything
    final = _candidates()[chosen]
    Xall, yall = _augment(ds_fit[FEATURES].to_numpy(float), ds_fit["y"].to_numpy(int))
    final.fit(Xall, yall)

    model_dir.mkdir(parents=True, exist_ok=True)
    report = TrainReport(chosen, results, len(ds))
    joblib.dump({"model": final, "name": chosen, "features": FEATURES},
                model_dir / MODEL_FILE)
    (model_dir / "train_report.json").write_text(json.dumps(asdict(report), indent=2))
    return report


def predict_upcoming(db: Database, model_dir: Path, save: bool = True) -> pd.DataFrame:
    """Return P(team A wins) for every upcoming fixture."""
    bundle = joblib.load(model_dir / MODEL_FILE)
    if bundle["features"] != FEATURES:
        raise RuntimeError("model was trained with a different feature set - retrain it")
    matches = db.matches_df()
    _, state = build_training_set(matches, db.rankings_df())
    fixtures = matches[matches["winner"].isna()]
    if fixtures.empty:
        return pd.DataFrame(columns=["match_id", "prob_a", "elo_prob"])
    fx = fixture_features(state, fixtures)
    fx["prob_a"] = symmetric_proba(bundle["model"], fx[FEATURES].to_numpy(float))
    if save:
        for r in fx.itertuples():
            db.save_prediction(int(r.match_id), bundle["name"], float(r.prob_a))
    return fx[["match_id", "date", "prob_a", "elo_prob"]]

"""Predictive check: how well can ANY model using the structured record anticipate later harm?

Complements the information analysis with a predictive test. Among the review candidates
(and, for context, among all affected bundles), later laboratory harm markers are predicted
from bundle-level features of the structured record with patient-grouped five-fold
cross-validation, so that no patient appears in both training and test folds.

Feature sets
  severity_only      highest base severity (what routing by severity alone sees)
  priority_score     CLARITY's review priority score (single feature)
  model_state        what the priority score reads, aggregated per bundle
  all_structured     model_state plus every further structured field in the replayed record
Models
  regularised logistic regression (always); gradient boosting (if scikit-learn is installed)
Reported: cross-validated area under the ROC curve (mean and range over folds) and the
Brier score. An area near 0.5 means no better than chance; modest gains from
all_structured over model_state support the claim that further structured fields add little.

Usage
  python experiments/predictive_check.py --project . --outputs outputs_corrected \
      --markers lab_harm_markers_<date>/private/bundle_markers.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = 5


def tf(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])


def auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p, kind="mergesort")
    r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    s = pd.Series(p); r = s.rank(method="average").to_numpy()
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logistic_fit_predict(Xtr, ytr, Xte, lam=1.0, iters=50):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]; B = np.c_[np.ones(len(Xte)), (Xte - mu) / sd]
    w = np.zeros(A.shape[1]); R = lam * np.eye(A.shape[1]); R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (p - ytr) + R @ w
        H = (A * (p * (1 - p))[:, None]).T @ A + R
        step = np.linalg.solve(H + 1e-8 * np.eye(len(w)), g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return 1 / (1 + np.exp(-np.clip(B @ w, -30, 30)))


def grouped_folds(groups: np.ndarray, k: int, seed: int = 20261031):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed); rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model: str):
    folds = grouped_folds(groups, FOLDS)
    aucs, briers = [], []
    for f in range(FOLDS):
        tr, te = folds != f, folds == f
        if y[te].sum() in (0, te.sum()):
            continue
        if model == "logistic":
            p = logistic_fit_predict(X[tr], y[tr], X[te])
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                               l2_regularization=1.0, random_state=0)
            m.fit(X[tr], y[tr]); p = m.predict_proba(X[te])[:, 1]
        aucs.append(auc(y[te], p)); briers.append(float(np.mean((p - y[te]) ** 2)))
    return (round(float(np.mean(aucs)), 3), round(float(np.min(aucs)), 3), round(float(np.max(aucs)), 3),
            round(float(np.mean(briers)), 4))


def bundle_features(dec: pd.DataFrame, extra: bool) -> pd.DataFrame:
    d = dec.copy()
    d["bundle_id"] = d["bundle_id"].astype(str)
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    d["_h"] = num("contextual_harm_index"); d["_e"] = num("evidence_support_index")
    d["_c"] = num("evidence_completeness_index"); d["_k"] = num("episode_occurrence_index")
    d["_sev"] = num("base_severity")
    g = d.groupby("bundle_id")
    F = pd.DataFrame({"n_concerns": g.size(), "harm_max": g["_h"].max(), "harm_mean": g["_h"].mean(),
                      "support_min": g["_e"].min(), "completeness_min": g["_c"].min(),
                      "occurrence_max": g["_k"].max(), "severity_max": g["_sev"].max()})
    cats = ["management_status", "repeat_status", "safety_class", "rule_family"]
    if extra:
        cats += ["transplant_phase", "protocol_fit", "monitoring_status", "transition_status"]
        for c, how in [("current_creatinine", "max"), ("hours_since_current_creatinine", "min"),
                       ("days_since_transplant", "min"), ("prior_lifetime_occurrences", "max"),
                       ("prior_admission_occurrences", "max"), ("hours_since_prior_same_signal", "min")]:
            if c in d:
                F[f"{c}_{how}"] = getattr(pd.to_numeric(d[c], errors="coerce").groupby(d["bundle_id"]), how)()
        for c in ["acute_renal_deterioration", "crrt_active", "material_change_since_prior"]:
            if c in d:
                F[f"any_{c}"] = tf(d[c]).groupby(d["bundle_id"]).max().astype(float)
    for c in cats:
        if c in d:
            dummies = pd.get_dummies(d[c].astype(str), prefix=c)
            F = F.join(dummies.groupby(d["bundle_id"]).max().astype(float))
    return F.fillna(F.median(numeric_only=True)).fillna(0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--outputs", default="outputs_corrected")
    ap.add_argument("--markers", type=Path, required=True)
    a = ap.parse_args()
    project = a.project.expanduser().resolve()
    private = project / a.outputs / "private"
    dec = pd.read_csv(private / "04_action_sets.csv", low_memory=False)
    bundles = pd.read_csv(private / "05_order_bundles.csv", low_memory=False)
    bundles["bundle_id"] = bundles["bundle_id"].astype(str)
    mk = pd.read_csv(a.markers, dtype={"bundle_id": str})
    y_all = mk.set_index("bundle_id")["any_marker"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    try:
        import sklearn  # noqa: F401
        models = ["logistic", "gradient_boosting"]
    except ImportError:
        models = ["logistic"]
        print("scikit-learn not installed: gradient boosting skipped")

    base = bundle_features(dec, extra=False)
    full = bundle_features(dec, extra=True)
    pool_mask = bundles["pre_capacity_action"].astype(str).eq("escalate") & ~tf(bundles["protected_alert"])
    populations = {"review_candidates": bundles.loc[pool_mask, "bundle_id"], "all_affected_bundles": bundles["bundle_id"]}

    out = project / ("predictive_check_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True)
    rows = []
    for pop, ids in populations.items():
        ids = pd.Index(ids).intersection(y_all.index)
        y = y_all.loc[ids].to_numpy()
        groups = bundles.set_index("bundle_id").loc[ids, "subject_id"].astype(str).to_numpy()
        sets = {
            "severity_only": bundles.set_index("bundle_id").loc[ids, ["highest_base_severity"]].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(),
            "priority_score": bundles.set_index("bundle_id").loc[ids, ["delta_expert"]].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(),
            "model_state": base.loc[ids].to_numpy(),
            "all_structured": full.loc[ids].to_numpy(),
        }
        for name, X in sets.items():
            for m in models:
                if m != "logistic" and X.shape[1] == 1:
                    continue
                t0 = time.time()
                mean, lo, hi, brier = evaluate(X.astype(float), y, groups, m)
                rows.append({"population": pop, "bundles": len(ids), "marker_rate_percent": round(100 * y.mean(), 1),
                             "features": name, "n_features": X.shape[1], "model": m,
                             "cv_auc_mean": mean, "cv_auc_min_fold": lo, "cv_auc_max_fold": hi, "cv_brier": brier})
                print(f"{pop} | {name} | {m}: AUC {mean} ({time.time() - t0:.0f}s)", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "predictive_check_summary.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({"folds": FOLDS, "grouping": "patient (subject_id)",
        "outcome": "any new-onset laboratory harm marker", "interpretation":
        "Markers reflect care received; AUC measures how well the recorded features anticipate them."}, indent=2))
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print("Saved:", out)


if __name__ == "__main__":
    main()

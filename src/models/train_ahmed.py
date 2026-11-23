"""
train_ahmed.py - Ahmed's Ridge regression + KNN (regression and classification) with baselines.

  Regression    : Ridge (main, alpha tuned by CV)  vs LinearRegression (baseline)
                  KNN regressor (tuned by CV)
  Classification: KNN (main, tuned by CV)          vs Logistic Regression (baseline)

Run from the project root:
    python -u -m src.models.train_ahmed            # fast run
    python -u -m src.models.train_ahmed --cv       # also reports 5-fold CV mean +/- std for Ridge
"""
import argparse
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.dataset import MODELS_DIR, RANDOM_STATE, RESULTS_DIR, get_split
from src.features.preprocessing import build_classification_pipeline, build_regression_pipeline
from src.models.metrics import (
    band_confusion,
    classification_metrics,
    error_by_group,
    regression_metrics,
)

AHMED_METRICS_PATH = Path(__file__).resolve().parents[2] / "reports" / "evidence" / "ahmed_metrics.csv"


def log_row(rows, task, name, split_name, metric_dict):
    rows.append({"task": task, "model": name, "set": split_name, **metric_dict})


def scaled(estimator_name, estimator):
    """Wrap an estimator with scaling. with_mean=False keeps it safe for sparse one-hot output."""
    return Pipeline([("scale", StandardScaler(with_mean=False)), (estimator_name, estimator)])


# ------------------------------------------------------------------ fitting helpers
def fit_regressor(rows, name, estimator, s, grid=None):
    """Fit a regression pipeline. If `grid` is given, tune it with CV on the train set only."""
    t0 = time.time()
    pipe = build_regression_pipeline(estimator)
    extra = {}
    if grid:
        search = GridSearchCV(pipe, grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
        search.fit(s.X_train, s.y_reg_train)
        pipe = search.best_estimator_
        extra = {"CV_MAE": -search.best_score_}
        print(f"      best params: {search.best_params_}")
    else:
        pipe.fit(s.X_train, s.y_reg_train)

    tr = regression_metrics(s.y_reg_train, pipe.predict(s.X_train))
    te = regression_metrics(s.y_reg_test, pipe.predict(s.X_test))
    log_row(rows, "regression", name, "train", tr)
    log_row(rows, "regression", name, "test", {**te, **extra})
    print(f"[REG] {name:<22} train R2={tr['R2']:.3f} | test R2={te['R2']:.3f} "
          f"| test MAE={te['MAE']:,.0f} | MAPE={te['MAPE_%']:.1f}% ({time.time()-t0:.0f}s)")
    return pipe


def fit_classifier(rows, name, estimator, s, scale=False, grid=None):
    """Fit a classification pipeline. If `grid` is given, tune it with CV on the train set only."""
    t0 = time.time()
    pipe = build_classification_pipeline(estimator, scale=scale)
    extra = {}
    if grid:
        search = GridSearchCV(pipe, grid, cv=3, scoring="f1_macro", n_jobs=-1)
        search.fit(s.X_train, s.y_clf_train)
        pipe = search.best_estimator_
        extra = {"CV_Macro_F1": search.best_score_}
        print(f"      best params: {search.best_params_}")
    else:
        pipe.fit(s.X_train, s.y_clf_train)

    tr = classification_metrics(s.y_clf_train, pipe.predict(s.X_train))
    te = classification_metrics(s.y_clf_test, pipe.predict(s.X_test))
    log_row(rows, "classification", name, "train", tr)
    log_row(rows, "classification", name, "test", {**te, **extra})
    print(f"[CLF] {name:<22} train acc={tr['Accuracy']:.3f} | test acc={te['Accuracy']:.3f} "
          f"| test macro-F1={te['Macro_F1']:.3f} ({time.time()-t0:.0f}s)")
    return pipe


def report_cv(rows, task, name, pipe_builder, X, y, scoring, k=5):
    """Optional: k-fold CV on the train set, to show variance across folds."""
    scores = cross_val_score(pipe_builder(), X, y, cv=k, scoring=scoring, n_jobs=-1)
    if scoring.startswith("neg_"):
        scores = -scores
    print(f"      {k}-fold CV {scoring}: {scores.mean():.3f} +/- {scores.std():.3f}")
    log_row(rows, task, name, f"cv{k}", {"CV_mean": scores.mean(), "CV_std": scores.std()})


# ------------------------------------------------------------------ plots / analysis
def ridge_coefficients(ridge_pipe):
    inner = ridge_pipe.regressor_                       # fitted Pipeline(pre -> model)
    names = inner.named_steps["pre"].get_feature_names_out()
    coef = inner.named_steps["model"].named_steps["ridge"].coef_
    return pd.Series(coef, index=names)


def plot_ridge_coefficients(ridge_pipe, top=20):
    coef = ridge_coefficients(ridge_pipe)
    alpha = ridge_pipe.regressor_.named_steps["model"].named_steps["ridge"].alpha
    print(f"      Ridge alpha={alpha} (shrinks coefficients but never sets them exactly to 0)")
    coef.rename("coef").sort_values(key=np.abs, ascending=False) \
        .to_csv(RESULTS_DIR / "ahmed_ridge_coefficients.csv")
    shown = coef.sort_values(key=np.abs).tail(top)
    plt.figure(figsize=(7, 6))
    shown.plot.barh()
    plt.title(f"Ridge - top {len(shown)} coefficients by size (scaled features)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ahmed_ridge_coefficients.png", dpi=150)
    plt.close()


def plot_permutation_importance(pipe, X, y, scoring, title, filename, top=15, n=1500, repeats=3):
    """Permutation importance on ORIGINAL columns, on a small test subsample (KNN predicts slowly)."""
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X), size=min(n, len(X)), replace=False)
    r = permutation_importance(pipe, X.iloc[idx], y.iloc[idx], scoring=scoring,
                               n_repeats=repeats, random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=X.columns).sort_values().tail(top)
    plt.figure(figsize=(7, 5))
    imp.plot.barh()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close()


def plot_confusion(cm_df, title, filename):
    cm = cm_df.values
    norm = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(len(cm_df)), cm_df.columns, rotation=30)
    plt.yticks(range(len(cm_df)), cm_df.index)
    for i in range(len(cm_df)):
        for j in range(len(cm_df)):
            colour = "white" if norm[i, j] > 0.5 else "black"
            plt.text(j, i, cm[i, j], ha="center", va="center", color=colour)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close()


# ------------------------------------------------------------------ main
def main(run_cv: bool):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    AHMED_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    s = get_split()
    print(f"Train {s.X_train.shape} | Test {s.X_test.shape}\n")

    # ================= REGRESSION =================
    fit_regressor(rows, "LinearRegression_baseline", LinearRegression(), s)

    ridge_pipe = fit_regressor(
        rows, "Ridge", scaled("ridge", Ridge(random_state=RANDOM_STATE)), s,
        grid={"regressor__model__ridge__alpha": [0.1, 1, 10, 100, 1000]})
    plot_ridge_coefficients(ridge_pipe)

    knn_reg = fit_regressor(
        rows, "KNN_reg", scaled("knn", KNeighborsRegressor(n_jobs=-1)), s,
        grid={"regressor__model__knn__n_neighbors": [3, 5, 7, 11, 15],
              "regressor__model__knn__weights": ["uniform", "distance"]})

    if run_cv:
        best_alpha = ridge_pipe.regressor_.named_steps["model"].named_steps["ridge"].alpha
        report_cv(rows, "regression", "Ridge",
                  lambda: build_regression_pipeline(
                      scaled("ridge", Ridge(alpha=best_alpha, random_state=RANDOM_STATE))),
                  s.X_train, s.y_reg_train, "neg_mean_absolute_error")

    # Error analysis for the main regressor (Ridge)
    pred = ridge_pipe.predict(s.X_test)
    error_by_group(s.y_reg_test, pred, s.y_clf_test).to_csv(RESULTS_DIR / "ahmed_ridge_error_by_band.csv")
    error_by_group(s.y_reg_test, pred, s.X_test["make"]).sort_values("count", ascending=False) \
        .head(15).to_csv(RESULTS_DIR / "ahmed_ridge_error_by_make.csv")
    plot_permutation_importance(knn_reg, s.X_test, s.y_reg_test, "neg_mean_absolute_error",
                                "KNN regressor - permutation importance (MAE)",
                                "ahmed_knn_reg_permutation_importance.png")

    joblib.dump(ridge_pipe, MODELS_DIR / "ahmed_ridge_regressor.joblib", compress=3)
    joblib.dump(knn_reg, MODELS_DIR / "ahmed_knn_regressor.joblib", compress=3)

    # ================= CLASSIFICATION =================
    print()
    knn_clf = fit_classifier(
        rows, "KNN_clf", KNeighborsClassifier(n_jobs=-1), s, scale=True,
        grid={"model__n_neighbors": [3, 5, 7, 11, 15],
              "model__weights": ["uniform", "distance"]})

    fit_classifier(
        rows, "LogReg_baseline",
        LogisticRegression(max_iter=2000, class_weight="balanced"), s, scale=True)

    cm = band_confusion(s.y_clf_test, knn_clf.predict(s.X_test))
    cm.to_csv(RESULTS_DIR / "ahmed_knn_clf_confusion_matrix.csv")
    plot_confusion(cm, "KNN - price band", "ahmed_knn_clf_confusion_matrix.png")
    plot_permutation_importance(knn_clf, s.X_test, s.y_clf_test, "f1_macro",
                                "KNN classifier - permutation importance (macro-F1)",
                                "ahmed_knn_clf_permutation_importance.png")

    joblib.dump(knn_clf, MODELS_DIR / "ahmed_knn_classifier.joblib", compress=3)

    # Save Ahmed's metrics without overwriting faiz_metrics.csv or farhan_metrics.csv
    pd.DataFrame(rows).to_csv(AHMED_METRICS_PATH, index=False)
    print(f"\nSaved: models/ahmed_*.joblib, {AHMED_METRICS_PATH}, and {RESULTS_DIR}/ahmed_*")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true", help="report 5-fold CV mean +/- std for Ridge")
    main(ap.parse_args().cv)
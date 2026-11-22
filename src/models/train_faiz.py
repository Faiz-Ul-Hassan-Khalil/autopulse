"""
train_faiz.py - Faiz's two models + baselines.

  Regression    : Random Forest (main)  vs Decision Tree (baseline)
  Classification: Decision Tree (main)  vs KNN (baseline)

Run from the project root:
    python -u -m src.models.train_faiz            # fast run with fixed settings
    python -u -m src.models.train_faiz --tune     # adds RandomizedSearchCV for the forest
"""
import argparse
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.features.dataset import MODELS_DIR, RANDOM_STATE, RESULTS_DIR, get_split
from src.features.preprocessing import build_classification_pipeline, build_regression_pipeline
from src.models.metrics import (
    band_confusion,
    classification_metrics,
    error_by_group,
    regression_metrics,
    save_metrics,
)

rows = []   # every result row ends up in reports/evidence/faiz_metrics.csv


def log_row(task, name, split_name, metric_dict):
    rows.append({"task": task, "model": name, "set": split_name, **metric_dict})


def fit_regressor(name, estimator, s):
    t0 = time.time()
    pipe = build_regression_pipeline(estimator)
    pipe.fit(s.X_train, s.y_reg_train)
    tr = regression_metrics(s.y_reg_train, pipe.predict(s.X_train))
    te = regression_metrics(s.y_reg_test, pipe.predict(s.X_test))
    log_row("regression", name, "train", tr)
    log_row("regression", name, "test", te)
    print(f"[REG] {name:<22} train R2={tr['R2']:.3f} | test R2={te['R2']:.3f} "
          f"| test MAE={te['MAE']:,.0f} | MAPE={te['MAPE_%']:.1f}% "
          f"({time.time()-t0:.0f}s)")
    return pipe


def fit_classifier(name, estimator, s, scale=False):
    t0 = time.time()
    pipe = build_classification_pipeline(estimator, scale=scale)
    pipe.fit(s.X_train, s.y_clf_train)
    tr = classification_metrics(s.y_clf_train, pipe.predict(s.X_train))
    te = classification_metrics(s.y_clf_test, pipe.predict(s.X_test))
    log_row("classification", name, "train", tr)
    log_row("classification", name, "test", te)
    print(f"[CLF] {name:<22} train acc={tr['Accuracy']:.3f} | test acc={te['Accuracy']:.3f} "
          f"| test macro-F1={te['Macro_F1']:.3f} ({time.time()-t0:.0f}s)")
    return pipe


def plot_importance(rf_pipe, top=15):
    inner = rf_pipe.regressor_                     # fitted Pipeline
    names = inner.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(inner.named_steps["model"].feature_importances_, index=names)
    imp = imp.sort_values().tail(top)
    plt.figure(figsize=(7, 5))
    imp.plot.barh()
    plt.title("Random Forest - top feature importances")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rf_feature_importance.png", dpi=150)
    plt.close()


def plot_confusion(cm_df, title, filename):
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm_df.values, cmap="Blues")
    plt.xticks(range(len(cm_df)), cm_df.columns, rotation=30)
    plt.yticks(range(len(cm_df)), cm_df.index)
    for i in range(len(cm_df)):
        for j in range(len(cm_df)):
            plt.text(j, i, cm_df.values[i, j], ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close()


def main(tune: bool):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    s = get_split()
    print(f"Train {s.X_train.shape} | Test {s.X_test.shape}\n")

    # ================= REGRESSION =================
    fit_regressor("DecisionTree_full",
                  DecisionTreeRegressor(random_state=RANDOM_STATE), s)       # overfits
    fit_regressor("DecisionTree_tuned",
                  DecisionTreeRegressor(max_depth=15, min_samples_leaf=10,
                                        random_state=RANDOM_STATE), s)

    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                               n_jobs=-1, random_state=RANDOM_STATE)
    rf_pipe = fit_regressor("RandomForest", rf, s)

    if tune:
        print("\nTuning Random Forest (RandomizedSearchCV, cv=3)...")
        search = RandomizedSearchCV(
            build_regression_pipeline(RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE)),
            param_distributions={
                "regressor__model__n_estimators": [150, 250, 350],
                "regressor__model__max_depth": [None, 20, 30],
                "regressor__model__min_samples_leaf": [1, 3, 5],
                "regressor__model__max_features": [0.5, 0.7, 1.0],
            },
            n_iter=8, cv=3, scoring="neg_mean_absolute_error",
            random_state=RANDOM_STATE, n_jobs=1, verbose=1)
        search.fit(s.X_train, s.y_reg_train)
        print("Best params:", search.best_params_)
        tuned = search.best_estimator_
        te = regression_metrics(s.y_reg_test, tuned.predict(s.X_test))
        log_row("regression", "RandomForest_tuned", "test", te)
        print(f"[REG] RandomForest_tuned     test R2={te['R2']:.3f} | MAE={te['MAE']:,.0f}")
        if te["MAE"] < regression_metrics(s.y_reg_test, rf_pipe.predict(s.X_test))["MAE"]:
            rf_pipe = tuned

    # error analysis + importance for the final forest
    pred = rf_pipe.predict(s.X_test)
    error_by_group(s.y_reg_test, pred, s.y_clf_test).to_csv(RESULTS_DIR / "rf_error_by_band.csv")
    error_by_group(s.y_reg_test, pred, s.X_test["make"]).sort_values("count", ascending=False) \
        .head(15).to_csv(RESULTS_DIR / "rf_error_by_make.csv")
    plot_importance(rf_pipe)
    joblib.dump(rf_pipe, MODELS_DIR / "rf_regressor.joblib", compress=3)

    # ================= CLASSIFICATION =================
    print()
    dt_clf = fit_classifier(
        "DecisionTree_clf",
        DecisionTreeClassifier(max_depth=12, min_samples_leaf=10,
                               class_weight="balanced", random_state=RANDOM_STATE), s)
    fit_classifier("KNN_baseline", KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
                   s, scale=True)

    cm = band_confusion(s.y_clf_test, dt_clf.predict(s.X_test))
    cm.to_csv(RESULTS_DIR / "dt_confusion_matrix.csv")
    plot_confusion(cm, "Decision Tree - price band", "dt_confusion_matrix.png")
    joblib.dump(dt_clf, MODELS_DIR / "dt_classifier.joblib", compress=3)

    save_metrics(rows)
    print(f"\nSaved: models/*.joblib and {RESULTS_DIR}/faiz_metrics.csv (+ plots)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    main(ap.parse_args().tune)
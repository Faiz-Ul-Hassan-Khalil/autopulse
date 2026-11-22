"""
preprocessing.py - shared encoding pipeline.

The encoders live INSIDE the sklearn Pipeline, so they are fitted on the
training rows only (no leakage into the test set).
"""
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

# Updated import to match your folder structure
from src.features.dataset import HIGH_CARD_COLS, LOW_CARD_COLS, RANDOM_STATE


def _target_encoder() -> TargetEncoder:
    """Works on scikit-learn 1.3 -> 1.9+ (random_state was deprecated in 1.9)."""
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 9):
        return TargetEncoder(cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))
    return TargetEncoder(random_state=RANDOM_STATE)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            # make / model / variant -> smoothed, cross-fitted target encoding
            ("hi_card", _target_encoder(), HIGH_CARD_COLS),
            # fuel_type / transmission -> one-hot
            ("lo_card", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             LOW_CARD_COLS),
        ],
        remainder="passthrough",      # numeric columns pass through unchanged
        verbose_feature_names_out=False,
    )


def _pipeline(estimator, scale: bool) -> Pipeline:
    steps = [("pre", build_preprocessor())]
    if scale:                         # only KNN / Logistic Regression need this
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def build_regression_pipeline(estimator, scale: bool = False):
    """Predicts log(price) internally, returns predictions in PKR."""
    return TransformedTargetRegressor(
        regressor=_pipeline(estimator, scale),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def build_classification_pipeline(estimator, scale: bool = False):
    return _pipeline(estimator, scale)
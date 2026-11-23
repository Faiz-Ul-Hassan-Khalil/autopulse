"""model_service.py - loads all 3 team members' saved models ONCE and serves predictions."""
import joblib
import pandas as pd

from backend.config import (
    DATA_PATH,
    DEAL_BELOW_PCT,
    ELECTRIC_FUEL,
    MODELS_DIR,
    OVERPRICED_ABOVE_PCT,
    RANGE_PCT,
    ROOT,
)
from backend.schemas import PredictRequest, PredictResponse, PriceRange, Verdict
from src.features.dataset import FEATURE_COLS, add_features

EVIDENCE_DIR = ROOT / "reports" / "evidence"

TEAM_MODELS = {
    "faiz": {
        "label": "Faiz — Random Forest + Decision Tree",
        "regressor": MODELS_DIR / "faiz_rf_regressor.joblib",
        "classifier": MODELS_DIR / "faiz_dt_classifier.joblib",
        "metrics": EVIDENCE_DIR / "faiz_metrics.csv",
    },
    "farhan": {
        "label": "Farhan — Decision Tree (Reg + Clf)",
        "regressor": MODELS_DIR / "farhan_dt_regressor.joblib",
        "classifier": MODELS_DIR / "farhan_dt_classifier.joblib",
        "metrics": EVIDENCE_DIR / "farhan_metrics.csv",
    },
    "ahmed": {
        "label": "Ahmed — KNN (Reg + Clf)",
        "regressor": MODELS_DIR / "ahmed_knn_regressor.joblib",
        "classifier": MODELS_DIR / "ahmed_knn_classifier.joblib",
        "metrics": EVIDENCE_DIR / "ahmed_metrics.csv",
    },
}


class ModelService:
    def __init__(self):
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Required dataset missing: {DATA_PATH}")

        self.bundles = {}
        for author, spec in TEAM_MODELS.items():
            for key in ("regressor", "classifier"):
                if not spec[key].exists():
                    raise FileNotFoundError(f"Missing {author} {key}: {spec[key]}")
            reg = joblib.load(spec["regressor"])
            clf = joblib.load(spec["classifier"])
            expected = list(reg.feature_names_in_)
            if expected != FEATURE_COLS:
                raise RuntimeError(
                    f"{author} model features differ from dataset.FEATURE_COLS.\n"
                    f"model: {expected}\ncode : {FEATURE_COLS}"
                )
            self.bundles[author] = {
                "label": spec["label"],
                "regressor": reg,
                "classifier": clf,
                "metrics_path": spec["metrics"],
            }

        self.catalog = self._build_catalog()

    def _build_catalog(self) -> dict:
        df = pd.read_csv(
            DATA_PATH,
            usecols=["make", "model", "variant", "engine_cc", "is_electric_unit"],
        )
        catalog = {}
        for make, g_make in df.groupby("make"):
            catalog[make] = {}
            for model, g_model in g_make.groupby("model"):
                non_ev = g_model.loc[g_model["is_electric_unit"] == 0, "engine_cc"]
                catalog[make][model] = {
                    "variants": sorted(g_model["variant"].unique().tolist()),
                    "typical_engine_cc": float(non_ev.median()) if len(non_ev) else None,
                }
        order = df["make"].value_counts().index.tolist()
        return {"makes": order, "models": catalog}

    def _canonical(self, value: str, options) -> str | None:
        v = value.strip().lower()
        return next((o for o in options if o.lower() == v), None)

    def predict(self, req: PredictRequest) -> PredictResponse:
        warnings = []
        cat = self.catalog["models"]

        make = self._canonical(req.make, cat.keys())
        model = self._canonical(req.model, cat[make].keys()) if make else None
        variant = (
            self._canonical(req.variant, cat[make][model]["variants"])
            if model
            else None
        )
        if make is None:
            warnings.append(
                f"Make '{req.make}' was not in the training data; the estimate is less reliable."
            )
        elif model is None:
            warnings.append(
                f"Model '{req.model}' was not seen for {make}; the estimate is less reliable."
            )
        elif variant is None:
            warnings.append(
                f"Variant '{req.variant}' was not seen for {make} {model}; the estimate is less reliable."
            )

        is_ev = req.fuel_type == ELECTRIC_FUEL
        row = pd.DataFrame([{
            "make": make or req.make.strip(),
            "model": model or req.model.strip(),
            "variant": variant or req.variant.strip(),
            "fuel_type": req.fuel_type,
            "transmission": req.transmission,
            "model_year": req.model_year,
            "mileage": req.mileage,
            "engine_cc": 0.0 if is_ev else float(req.engine_cc),
            "battery_kwh": float(req.battery_kwh) if is_ev else 0.0,
            "is_electric_unit": int(is_ev),
            "has_auction_rating": int(req.auction_rating is not None),
            "auction_rating_filled": float(req.auction_rating or 0.0),
        }])
        X = add_features(row)[FEATURE_COLS]

        bundle = self.bundles[req.author]
        price = float(bundle["regressor"].predict(X)[0])
        band = str(bundle["classifier"].predict(X)[0])

        resp = PredictResponse(
            author=req.author,
            model_label=bundle["label"],
            predicted_price_pkr=round(price, -3),
            price_range=PriceRange(
                low=round(price * (1 - RANGE_PCT), -3),
                high=round(price * (1 + RANGE_PCT), -3),
            ),
            price_band=band,
            warnings=warnings,
        )
        if req.listed_price_pkr is not None:
            resp.verdict = self._verdict(req.listed_price_pkr, price)
        return resp

    @staticmethod
    def _verdict(listed: float, predicted: float) -> Verdict:
        diff = (listed - predicted) / predicted * 100
        if diff > OVERPRICED_ABOVE_PCT:
            return Verdict(
                status="overpriced",
                difference_pct=round(diff, 1),
                message=f"Listed {diff:.0f}% above the estimated fair price.",
            )
        if diff < DEAL_BELOW_PCT:
            return Verdict(
                status="possible_deal",
                difference_pct=round(diff, 1),
                message=f"Listed {abs(diff):.0f}% below the estimate. Verify the car's condition before buying.",
            )
        return Verdict(
            status="fair",
            difference_pct=round(diff, 1),
            message="Listed price is close to the estimated fair price.",
        )

    def model_info(self) -> dict:
        team_metrics = {}
        all_rows = []
        for author, bundle in self.bundles.items():
            p = bundle["metrics_path"]
            if p.exists():
                m = pd.read_csv(p)
                records = m.astype(object).where(pd.notnull(m), None).to_dict(orient="records")
                team_metrics[author] = {
                    "label": bundle["label"],
                    "metrics": records,
                }
                for r in records:
                    all_rows.append({"author": author, **r})
        return {
            "available": len(all_rows) > 0,
            "team": team_metrics,
            "metrics": all_rows,
        }
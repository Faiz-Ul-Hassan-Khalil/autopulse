"""model_service.py - loads the saved models ONCE and turns requests into predictions."""
import joblib
import pandas as pd

from backend.config import (
    DATA_PATH,
    DEAL_BELOW_PCT,
    DT_PATH,
    ELECTRIC_FUEL,
    METRICS_PATH,
    OVERPRICED_ABOVE_PCT,
    RANGE_PCT,
    RF_PATH,
)
from backend.schemas import PredictRequest, PredictResponse, PriceRange, Verdict
# Same feature code as training -> no train/serve mismatch.
from src.features.dataset import FEATURE_COLS, add_features


class ModelService:
    def __init__(self):
        for p in (RF_PATH, DT_PATH, DATA_PATH):
            if not p.exists():
                raise FileNotFoundError(f"Required file missing: {p}")
        self.regressor = joblib.load(RF_PATH)
        self.classifier = joblib.load(DT_PATH)
        expected = list(self.regressor.feature_names_in_)
        if expected != FEATURE_COLS:
            raise RuntimeError(
                "Saved model features differ from dataset.FEATURE_COLS.\n"
                f"model: {expected}\ncode : {FEATURE_COLS}\n"
                "Retrain the models or fix FEATURE_COLS."
            )
        self.catalog = self._build_catalog()

    # ---------- dropdown data for the frontend form ----------
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
        order = df["make"].value_counts().index.tolist()      # most common brands first
        return {"makes": order, "models": catalog}

    def _canonical(self, value: str, options) -> str | None:
        v = value.strip().lower()
        return next((o for o in options if o.lower() == v), None)

    # ---------- prediction ----------
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

        price = float(self.regressor.predict(X)[0])
        band = str(self.classifier.predict(X)[0])
        resp = PredictResponse(
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
        if not METRICS_PATH.exists():
            return {"available": False}
        m = pd.read_csv(METRICS_PATH)
        return {"available": True, "metrics": m.to_dict(orient="records")}
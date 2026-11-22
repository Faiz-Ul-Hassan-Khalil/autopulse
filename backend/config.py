"""config.py - all backend settings in one place."""
from pathlib import Path

# parents[1] because this lives in autopulse/backend/config.py
ROOT = Path(__file__).resolve().parents[1]            # project root (autopulse/)
DATA_PATH = ROOT / "data" / "processed" / "pakistan_cars_clean.csv"
MODELS_DIR = ROOT / "models"
RF_PATH = MODELS_DIR / "rf_regressor.joblib"          # price regression model
DT_PATH = MODELS_DIR / "dt_classifier.joblib"         # price band classifier
METRICS_PATH = ROOT / "reports" / "evidence" / "faiz_metrics.csv"

# Allowed values (must match the training data spelling)
FUEL_TYPES = ["Petrol", "Hybrid", "Diesel", "Electric", "PHEV", "CNG", "LPG", "REEV"]
TRANSMISSIONS = ["Automatic", "Manual"]
ELECTRIC_FUEL = "Electric"      # fuel type that uses battery_kwh instead of engine_cc

# Price range shown to the user = prediction +/- RANGE_PCT (measured test MAPE ~9.9%)
RANGE_PCT = 0.10

# Fair-price verdict thresholds (listed price vs predicted price, in %)
OVERPRICED_ABOVE_PCT = 15
DEAL_BELOW_PCT = -15

# Frontend dev servers & mobile clients allowed to call this API
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8081",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "*",
]
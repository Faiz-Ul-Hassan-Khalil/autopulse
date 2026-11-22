"""schemas.py - request / response shapes (Pydantic validates every input)."""
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.config import ELECTRIC_FUEL, FUEL_TYPES, TRANSMISSIONS
from src.features.dataset import CURRENT_YEAR


class PredictRequest(BaseModel):
    # model_year starts with "model_", which pydantic reserves by default
    model_config = ConfigDict(protected_namespaces=())

    make: str = Field(..., min_length=1, examples=["Toyota"])
    model: str = Field(..., min_length=1, examples=["Corolla"])
    variant: str = Field(..., min_length=1, examples=["GLi 1.3"])
    model_year: int = Field(..., ge=1980, le=CURRENT_YEAR, examples=[2020])
    mileage: int = Field(..., ge=0, le=1_000_000, description="odometer in km", examples=[55000])
    fuel_type: str = Field(..., examples=["Petrol"])
    transmission: str = Field(..., examples=["Automatic"])
    engine_cc: Optional[float] = Field(None, ge=600, le=7000, examples=[1300])
    battery_kwh: Optional[float] = Field(None, ge=15, le=200, examples=[None])
    auction_rating: Optional[float] = Field(None, ge=1.0, le=10.0, examples=[None])
    listed_price_pkr: Optional[float] = Field(None, gt=0, description="seller's asking price", examples=[3800000])

    @model_validator(mode="after")
    def check_combination(self):
        fuel = next((f for f in FUEL_TYPES if f.lower() == self.fuel_type.strip().lower()), None)
        if fuel is None:
            raise ValueError(f"fuel_type must be one of {FUEL_TYPES}")
        trans = next((t for t in TRANSMISSIONS if t.lower() == self.transmission.strip().lower()), None)
        if trans is None:
            raise ValueError(f"transmission must be one of {TRANSMISSIONS}")
        self.fuel_type, self.transmission = fuel, trans
        if fuel == ELECTRIC_FUEL and self.battery_kwh is None:
            raise ValueError("battery_kwh is required for Electric cars")
        if fuel != ELECTRIC_FUEL and self.engine_cc is None:
            raise ValueError("engine_cc is required for non-electric cars")
        return self


class PriceRange(BaseModel):
    low: int
    high: int


class Verdict(BaseModel):
    status: str                  # "fair" | "overpriced" | "possible_deal"
    difference_pct: float        # (listed - predicted) / predicted * 100
    message: str


class PredictResponse(BaseModel):
    predicted_price_pkr: int
    price_range: PriceRange
    price_band: str
    verdict: Optional[Verdict] = None
    warnings: list[str] = []


class DropdownOptionsResponse(BaseModel):
    makes: List[str]
    models_by_make: Dict[str, List[str]]
    variants_by_model: Dict[str, List[str]]
    fuel_types: List[str]
    transmissions: List[str]
    year_range: Dict[str, int]
from pathlib import Path
import re
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "data" / "raw").exists():
    PROJECT_ROOT = Path.cwd()

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Multi-word car makes present in Pakistani market listings
MULTI_WORD_MAKES = {
    "alfa romeo": "Alfa Romeo",
    "aston martin": "Aston Martin",
    "great wall": "Great Wall",
    "land rover": "Land Rover",
    "mercedes benz": "Mercedes-Benz",
    "mercedes-benz": "Mercedes-Benz",
    "range rover": "Range Rover",
    "rolls royce": "Rolls-Royce",
    "rolls-royce": "Rolls-Royce",
}


def load_raw(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def parse_car_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits '<Make> <Model>  <Model Year> <Variant> for Sale' into structured columns
    using Model Year as the anchor (verified 100% match rate in raw audit).
    """
    makes, models, variants = [], [], []

    for name, year in zip(df["Car Name"].astype(str), df["Model Year"].astype(int)):
        anchor = f" {year} "
        if anchor in name:
            left, right = name.split(anchor, 1)
        else:
            left, right = name, ""

        left_clean = " ".join(left.strip().split())
        right_clean = re.sub(r"\s+for\s+Sale\s*$", "", right.strip(), flags=re.IGNORECASE).strip()

        # Extract Make (handling multi-word makes first) and Model
        words = left_clean.split(" ")
        first_two = " ".join(words[:2]).lower() if len(words) >= 2 else ""

        if first_two in MULTI_WORD_MAKES:
            make = MULTI_WORD_MAKES[first_two]
            model = " ".join(words[2:]).strip() or "Unknown"
        else:
            make = words[0] if words else "Unknown"
            model = " ".join(words[1:]).strip() or "Unknown"

        variant = right_clean if right_clean else "Standard"

        makes.append(make)
        models.append(model)
        variants.append(variant)

    df["make"] = makes
    df["model"] = models
    df["variant"] = variants
    return df


def clean_pakwheels(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 75)
    print("CLEANING MAIN DATASET: Pakistan_Used_Cars.csv")
    print("=" * 75)
    initial_rows = len(df)
    print(f"Initial raw rows: {initial_rows:,}")

    # 1. Strip whitespace in string columns
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # 2. Drop rows missing the target Price (PKR)
    df["Price (PKR)"] = pd.to_numeric(df["Price (PKR)"], errors="coerce")
    df = df.dropna(subset=["Price (PKR)"]).copy()
    print(f"After dropping null Price (PKR): {len(df):,} (-{initial_rows - len(df):,})")

    # 3. Deduplicate on listing attributes (ignoring Auction Rating so duplicate ads merge cleanly)
    before_dedup = len(df)
    listing_cols = [c for c in df.columns if c != "Auction Rating"]
    # Sort so rows WITH a valid Auction Rating are kept first when deduplicating
    df = df.sort_values(by="Auction Rating", na_position="last")
    df = df.drop_duplicates(subset=listing_cols, keep="first").copy()
    print(f"After removing exact/near duplicates: {len(df):,} (-{before_dedup - len(df):,})")

    # 4. Filter invalid Model Years (< 1980 are vintage/placeholder anomalies)
    before_year = len(df)
    df = df[(df["Model Year"] >= 1980) & (df["Model Year"] <= 2026)].copy()
    print(f"After filtering Model Year [1980..2026]: {len(df):,} (-{before_year - len(df):,})")

    # 5. Clean Engine Capacity separately by Engine Unit (cc vs kWh)
    before_eng = len(df)
    is_kwh = df["Engine Unit"].str.lower() == "kwh"

    # Fix obvious decimal-omission typos in kWh (e.g., 810 -> 81.0, 9600 -> 96.0)
    kwh_vals = df.loc[is_kwh, "Engine Capacity"].copy()
    kwh_vals = np.where(kwh_vals > 1000, kwh_vals / 100.0, kwh_vals)
    kwh_vals = np.where(kwh_vals > 250, kwh_vals / 10.0, kwh_vals)
    df.loc[is_kwh, "Engine Capacity"] = kwh_vals

    # Keep realistic passenger car cc [600..7000] and EV battery kWh [15..200]
    valid_cc = (~is_kwh) & (df["Engine Capacity"] >= 600) & (df["Engine Capacity"] <= 7000)
    valid_kwh = is_kwh & (df["Engine Capacity"] >= 15) & (df["Engine Capacity"] <= 200)
    df = df[valid_cc | valid_kwh].copy()
    print(f"After cleaning Engine Capacity bounds (cc [600..7000], kWh [15..200]): {len(df):,} (-{before_eng - len(df):,})")

    # 6. Remove extreme price typos (keep [200,000 .. 150,000,000] PKR = 2 Lac to 15 Crore)
    before_price = len(df)
    df = df[(df["Price (PKR)"] >= 200_000) & (df["Price (PKR)"] <= 150_000_000)].copy()
    print(f"After filtering extreme Price typos [200K..150M PKR]: {len(df):,} (-{before_price - len(df):,})")

    # 7. Parse Car Name into make, model, variant
    df = parse_car_name(df)

    # 8. Engineer clean domain columns (row-level, zero leakage)
    df["is_electric_unit"] = (df["Engine Unit"].str.lower() == "kwh").astype(int)
    df["engine_cc"] = np.where(df["is_electric_unit"] == 0, df["Engine Capacity"], 0.0)
    df["battery_kwh"] = np.where(df["is_electric_unit"] == 1, df["Engine Capacity"], 0.0)

    df["has_auction_rating"] = df["Auction Rating"].notna().astype(int)
    df["auction_rating_filled"] = df["Auction Rating"].fillna(0.0)

    # Standardize column names to snake_case
    df = df.rename(columns={
        "Model Year": "model_year",
        "Mileage": "mileage",
        "Fuel Type": "fuel_type",
        "Transmission": "transmission",
        "Price (PKR)": "price_pkr",
    })

    keep_cols = [
        "make", "model", "variant", "model_year", "mileage",
        "fuel_type", "transmission", "is_electric_unit",
        "engine_cc", "battery_kwh", "has_auction_rating",
        "auction_rating_filled", "price_pkr"
    ]
    df = df[keep_cols].reset_index(drop=True)

    print(f"\nFINAL CLEAN PAKWHEELS SHAPE: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"Unique Makes: {df['make'].nunique()} | Unique Models: {df['model'].nunique()} | Unique Variants: {df['variant'].nunique()}")
    print("\nTop 15 Makes after parsing:")
    print(df["make"].value_counts().head(15).to_string())
    return df


def clean_olx(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 75)
    print("CLEANING NLP DATASET: olx_used_cars.csv")
    print("=" * 75)
    initial_rows = len(df)
    print(f"Initial raw rows: {initial_rows:,}")

    # Drop zero-variance 'Condition' and scraping URL columns
    drop_cols = [c for c in ["Condition", "Images URL's", "Car Profile"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Deduplicate
    df = df.drop_duplicates().copy()
    print(f"After removing exact duplicates: {len(df):,} (-{initial_rows - len(df):,})")

    # Clean phone numbers and excessive whitespace from Description for clean NLP
    def clean_text(text: str) -> str:
        text = str(text)
        # Mask Pakistani mobile numbers (e.g. 0300-1234567 or 0/3/3/2/...)
        text = re.sub(r"0[\s/\-]*3[\d\s/\-]{8,15}", " <PHONE> ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df["description_clean"] = df["Description"].apply(clean_text)

    df = df.rename(columns={
        "Price": "price_pkr",
        "Make": "make",
        "Model": "model",
        "Year": "model_year",
        "KM's driven": "mileage",
        "Fuel": "fuel_type",
        "Registration city": "registration_city",
        "Car documents": "car_documents",
        "Assembly": "assembly",
        "Transmission": "transmission",
        "Seller Location": "seller_location",
        "Car Features": "car_features",
    }).drop(columns=["Description"]).reset_index(drop=True)

    print(f"FINAL CLEAN OLX SHAPE: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


if __name__ == "__main__":
    df_main_raw = load_raw("Pakistan_Used_Cars.csv")
    df_main_clean = clean_pakwheels(df_main_raw)
    df_main_clean.to_csv(PROCESSED_DIR / "pakistan_cars_clean.csv", index=False)

    df_olx_raw = load_raw("olx_used_cars.csv")
    df_olx_clean = clean_olx(df_olx_raw)
    df_olx_clean.to_csv(PROCESSED_DIR / "olx_clean.csv", index=False)

    print("\nSaved cleaned datasets to data/processed/ successfully.")
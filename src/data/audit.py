from pathlib import Path
import numpy as np
import pandas as pd

# Resolve project root safely whether run from root or src/data/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "data" / "raw").exists():
    PROJECT_ROOT = Path.cwd()

RAW_DIR = PROJECT_ROOT / "data" / "raw"

FILES = {
    "main_pakwheels": RAW_DIR / "Pakistan_Used_Cars.csv",
    "bmw_pakwheels": RAW_DIR / "BMW_Cars_Pakistan.csv",
    "olx": RAW_DIR / "olx_used_cars.csv",
}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Check that CSVs are inside data/raw/.")

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        print(f"[{path.name}] Dropping index-like columns: {unnamed}")
        df = df.drop(columns=unnamed)

    return df


def audit_dataframe(name: str, df: pd.DataFrame, price_col: str) -> None:
    print("=" * 75)
    print(f"DATASET: {name} | Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"Columns: {df.columns.tolist()}")
    print("=" * 75)

    # 1. Column types, missingness & cardinality
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "null_count": df.isna().sum(),
        "null_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(dropna=True),
    })
    print("\n--- Column Summary ---")
    print(summary.to_string())

    # 2. Numeric summary (catches impossible years, mileage, engine sizes)
    num_desc = df.describe(include="number").T
    if not num_desc.empty:
        print("\n--- Numeric Summary ---")
        print(num_desc.to_string())

    # 3. Exact full-row duplicates (after dropping Unnamed index columns)
    exact_dups = df.duplicated().sum()
    print(f"\nExact full-row duplicates: {exact_dups:,} ({exact_dups / len(df) * 100:.2f}%)")

    # 4. Price parsing & quantiles
    raw_price = df[price_col]
    prices = pd.to_numeric(raw_price, errors="coerce")
    coerced_bad = int(prices.isna().sum() - raw_price.isna().sum())
    print(f"\nNon-numeric price strings coerced to NaN: {coerced_bad:,}")
    if coerced_bad > 0:
        bad_samples = raw_price[raw_price.notna() & prices.isna()].unique()[:10]
        print(f"Sample non-numeric price values: {bad_samples}")

    valid_prices = prices.dropna()
    quantiles = valid_prices.quantile([0.0, 0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995, 1.0])
    print(f"\n--- Price Quantiles ({price_col}) [Valid numeric: {len(valid_prices):,}] ---")
    for q, val in quantiles.items():
        print(f"  {q*100:5.1f}% : {val:,.0f}")
    print(f"  Zero or negative prices: {(valid_prices <= 0).sum():,}\n")


def check_pakwheels_specifics(df_main: pd.DataFrame, df_bmw: pd.DataFrame) -> None:
    print("=" * 75)
    print("PAKWHEELS SPECIFIC CHECKS (MAIN vs BMW)")
    print("=" * 75)

    # Overlap check without NaN==NaN false matches
    key_cols = ["Car Name", "Model Year", "Mileage", "Price (PKR)"]
    bmw_k = df_bmw.dropna(subset=key_cols).drop_duplicates(subset=key_cols)
    main_k = df_main.dropna(subset=key_cols).drop_duplicates(subset=key_cols)
    overlap = bmw_k.merge(main_k[key_cols], on=key_cols, how="inner")

    print(f"BMW rows with valid keys: {len(bmw_k)} / {len(df_bmw)}")
    print(f"BMW rows (valid keys) also in Main: {len(overlap)} / {len(bmw_k)}")
    print(f"BMW cars inside Main by name prefix: {df_main['Car Name'].str.startswith('BMW', na=False).sum():,}")

    # Near-duplicates in main dataset (ignoring Auction Rating)
    listing_cols = [c for c in df_main.columns if c != "Auction Rating"]
    near_dups = df_main.duplicated(subset=listing_cols).sum()
    print(f"Main near-duplicates (all cols except Auction Rating): {near_dups:,}")

    # Engine Unit & Fuel Type breakdown
    print("\n--- Engine Unit Counts (Main) ---")
    print(df_main["Engine Unit"].value_counts(dropna=False).to_string())

    print("\n--- Fuel Type Counts (Main) ---")
    print(df_main["Fuel Type"].value_counts(dropna=False).to_string())

    # Engine Capacity ranges by Engine Unit
    print("\n--- Engine Capacity Min/Median/Max by Engine Unit ---")
    print(
        df_main.groupby("Engine Unit", dropna=False)["Engine Capacity"]
        .agg(["count", "min", "median", "max"])
        .to_string()
    )

    # Safe year-anchor check (handles float years when NaNs exist)
    years = pd.to_numeric(df_main["Model Year"], errors="coerce")
    has_year_anchor = np.mean([
        (not np.isnan(y)) and (f" {int(y)} " in str(name))
        for y, name in zip(years, df_main["Car Name"])
    ]) * 100
    print(f"\nRows where ' <Model Year> ' cleanly splits Car Name: {has_year_anchor:.2f}%\n")


def check_olx_specifics(df_olx: pd.DataFrame) -> None:
    print("=" * 75)
    print("OLX SPECIFIC CHECKS")
    print("=" * 75)
    for col in ["Fuel", "Assembly", "Transmission", "Condition", "Car documents"]:
        if col in df_olx.columns:
            print(f"\n--- {col} ---")
            print(df_olx[col].value_counts(dropna=False).to_string())

    if "Car Features" in df_olx.columns:
        non_null_feats = df_olx["Car Features"].dropna()
        unique_tags = set()
        for entry in non_null_feats:
            for tag in str(entry).split(","):
                t = tag.strip()
                if t:
                    unique_tags.add(t)
        print(f"\nUnique comma-separated tags in 'Car Features': {len(unique_tags)}")
        print("Sample tags:", sorted(unique_tags)[:15])


if __name__ == "__main__":
    dfs = {k: load_csv(path) for k, path in FILES.items()}

    audit_dataframe("Pakistan_Used_Cars.csv", dfs["main_pakwheels"], "Price (PKR)")
    audit_dataframe("BMW_Cars_Pakistan.csv", dfs["bmw_pakwheels"], "Price (PKR)")
    audit_dataframe("olx_used_cars.csv", dfs["olx"], "Price")

    check_pakwheels_specifics(dfs["main_pakwheels"], dfs["bmw_pakwheels"])
    check_olx_specifics(dfs["olx"])
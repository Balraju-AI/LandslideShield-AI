import os
import numpy as np
import pandas as pd


INPUT_FILE = "data/india_landslides.csv"
OUTPUT_FILE = "data/processed_landslides.csv"


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


def main():

    print("=" * 70)
    print("LANDSLIDESHIELD AI - DATA PREPROCESSING")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    # Clean column names
    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    print("\nOriginal shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required = [
        "event_date",
        "latitude",
        "longitude",
        "landslide_trigger"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: " +
            ", ".join(missing)
        )

    # -----------------------------------------------------
    # State
    # -----------------------------------------------------

    if "state" not in df.columns:
        df["state"] = "Unknown"

    df["state"] = (
        df["state"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    # -----------------------------------------------------
    # Date
    # -----------------------------------------------------

    df["event_date"] = pd.to_datetime(
        df["event_date"],
        errors="coerce"
    )

    # -----------------------------------------------------
    # Coordinates
    # -----------------------------------------------------

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------

    df["landslide_trigger"] = (
        df["landslide_trigger"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # -----------------------------------------------------
    # Remove invalid records
    # -----------------------------------------------------

    df = df.dropna(
        subset=[
            "event_date",
            "latitude",
            "longitude"
        ]
    ).copy()

    # -----------------------------------------------------
    # Date features
    # -----------------------------------------------------

    df["year"] = (
        df["event_date"]
        .dt.year
        .astype(int)
    )

    df["month"] = (
        df["event_date"]
        .dt.month
        .astype(int)
    )

    df["day"] = (
        df["event_date"]
        .dt.day
        .astype(int)
    )

    df["day_of_year"] = (
        df["event_date"]
        .dt.dayofyear
        .astype(int)
    )

    # Cyclic month features
    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # Season
    df["season"] = df["month"].apply(
        get_season
    )

    # -----------------------------------------------------
    # Sort
    # -----------------------------------------------------

    df = df.sort_values(
        "event_date"
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    os.makedirs(
        "data",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nProcessed shape:")
    print(df.shape)

    print("\nTrigger distribution:")
    print(
        df["landslide_trigger"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print("\nPreprocessing completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
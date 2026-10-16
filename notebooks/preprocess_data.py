import os
import pandas as pd

INPUT_FILE = "data/india_landslides.csv"
OUTPUT_FILE = "data/processed_landslides.csv"


REQUIRED_COLUMNS = [
    "event_date",
    "latitude",
    "longitude",
    "landslide_trigger"
]


def clean_trigger(value):
    if pd.isna(value):
        return "unknown"

    text = str(value).strip().lower()

    if text == "":
        return "unknown"

    text = text.replace("-", "_")
    text = text.replace(" ", "_")

    return text


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5]:
        return "Summer"

    if month in [6, 7, 8, 9]:
        return "Monsoon"

    return "Post-Monsoon"


def main():

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    print("=" * 60)
    print("LANDSLIDESHIELD AI - DATA PREPROCESSING")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nOriginal records: {len(df):,}")

    # -----------------------------------------------------
    # Check columns
    # -----------------------------------------------------

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
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
    # Trigger
    # -----------------------------------------------------

    df["landslide_trigger"] = (
        df["landslide_trigger"]
        .apply(clean_trigger)
    )

    # -----------------------------------------------------
    # Remove invalid records
    # -----------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            "event_date",
            "latitude",
            "longitude"
        ]
    ).copy()

    print(
        f"Removed invalid records: "
        f"{before - len(df):,}"
    )

    # -----------------------------------------------------
    # Date features
    # -----------------------------------------------------

    df["year"] = df["event_date"].dt.year

    df["month"] = df["event_date"].dt.month

    df["day"] = df["event_date"].dt.day

    df["day_of_year"] = (
        df["event_date"].dt.dayofyear
    )

    df["month_sin"] = (
        __import__("numpy").sin(
            2 * __import__("numpy").pi *
            df["month"] / 12
        )
    )

    df["month_cos"] = (
        __import__("numpy").cos(
            2 * __import__("numpy").pi *
            df["month"] / 12
        )
    )

    df["season"] = df["month"].apply(
        get_season
    )

    # -----------------------------------------------------
    # Keep useful columns
    # -----------------------------------------------------

    output_columns = [
        "event_date",
        "latitude",
        "longitude",
        "year",
        "month",
        "day",
        "day_of_year",
        "month_sin",
        "month_cos",
        "season",
        "landslide_trigger"
    ]

    df = df[output_columns]

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

    print(
        f"\nProcessed records: {len(df):,}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("\nTrigger distribution:")

    print(
        df["landslide_trigger"]
        .value_counts()
        .to_string()
    )

    print("\nPreprocessing completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
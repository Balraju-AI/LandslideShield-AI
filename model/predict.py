import os
import joblib
import pandas as pd
from datetime import date


MODEL_FILE = "model/landslide_model.pkl"


def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5]:
        return "Summer"

    if month in [6, 7, 8, 9]:
        return "Monsoon"

    return "Post-Monsoon"


def build_input(
    latitude,
    longitude,
    selected_date
):

    month = selected_date.month

    day = selected_date.day

    year = selected_date.year

    day_of_year = (
        selected_date.timetuple().tm_yday
    )

    import numpy as np

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    season = get_season(month)

    return pd.DataFrame([{
        "latitude": latitude,
        "longitude": longitude,
        "year": year,
        "month": month,
        "day": day,
        "day_of_year": day_of_year,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "season": season
    }])


def main():

    if not os.path.exists(MODEL_FILE):

        print(
            "Model not found."
        )

        print(
            "Run train_model.py first."
        )

        return

    bundle = joblib.load(
        MODEL_FILE
    )

    model = bundle["model"]

    print("=" * 55)
    print("LANDSLIDESHIELD AI - TEST PREDICTION")
    print("=" * 55)

    latitude = float(
        input("Latitude: ")
    )

    longitude = float(
        input("Longitude: ")
    )

    year = int(
        input("Year: ")
    )

    month = int(
        input("Month (1-12): ")
    )

    day = int(
        input("Day: ")
    )

    selected_date = date(
        year,
        month,
        day
    )

    data = build_input(
        latitude,
        longitude,
        selected_date
    )

    prediction = model.predict(
        data
    )[0]

    probabilities = model.predict_proba(
        data
    )[0]

    confidence = (
        probabilities.max() * 100
    )

    print("\n" + "=" * 55)

    print("PREDICTION RESULT")

    print("=" * 55)

    print(
        f"Predicted trigger : {prediction}"
    )

    print(
        f"AI confidence     : {confidence:.2f}%"
    )

    print(
        f"Season             : {get_season(month)}"
    )

    print("=" * 55)


if __name__ == "__main__":
    main()
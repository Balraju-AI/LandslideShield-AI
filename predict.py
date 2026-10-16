import os
import joblib
import pandas as pd
import numpy as np
from datetime import date


MODEL_PATH = "model/landslide_model.pkl"


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

    print("\n" + "=" * 70)
    print("        LANDSLIDESHIELD AI")
    print("        LANDSLIDE PREDICTION SYSTEM")
    print("=" * 70)

    # ---------------------------------------------------------
    # CHECK MODEL
    # ---------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        print("\nERROR: Model file not found.")
        print("Expected:", MODEL_PATH)
        print("\nRun:")
        print("python train_model.py")
        return

    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------

    try:

        model = joblib.load(MODEL_PATH)

    except Exception as e:

        print("\nERROR: Could not load model.")
        print(e)
        return

    print("\nAI model loaded successfully.")

    # ---------------------------------------------------------
    # USER INPUT
    # ---------------------------------------------------------

    try:

        print("\n" + "-" * 70)
        print("LOCATION")
        print("-" * 70)

        latitude = float(
            input("Latitude: ")
        )

        longitude = float(
            input("Longitude: ")
        )

        state = input(
            "State: "
        ).strip()

        print("\n" + "-" * 70)
        print("DATE")
        print("-" * 70)

        year = int(
            input("Year: ")
        )

        month = int(
            input("Month (1-12): ")
        )

        day = int(
            input("Day (1-31): ")
        )

        # Validate date

        selected_date = date(
            year,
            month,
            day
        )

    except ValueError as e:

        print("\n" + "=" * 70)
        print("INVALID INPUT")
        print("=" * 70)

        print("\nPlease enter numbers correctly.")

        print("\nExample:")

        print("Latitude: 17.3850")
        print("Longitude: 78.4867")
        print("State: Telangana")
        print("Year: 2026")
        print("Month: 9")
        print("Day: 2")

        print("\nError:", e)

        return

    # ---------------------------------------------------------
    # DATE FEATURES
    # ---------------------------------------------------------

    day_of_year = (
        selected_date
        .timetuple()
        .tm_yday
    )

    # ---------------------------------------------------------
    # SEASON
    # ---------------------------------------------------------

    season = get_season(month)

    # ---------------------------------------------------------
    # CYCLICAL MONTH FEATURES
    #
    # IMPORTANT:
    # The saved model expects these.
    # ---------------------------------------------------------

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    # ---------------------------------------------------------
    # CREATE MODEL INPUT
    # ---------------------------------------------------------

    input_data = pd.DataFrame([{

        "latitude": latitude,

        "longitude": longitude,

        "year": year,

        "month": month,

        "day": day,

        "day_of_year": day_of_year,

        "season": season,

        "state": state,

        "month_sin": month_sin,

        "month_cos": month_cos

    }])

    # ---------------------------------------------------------
    # DISPLAY INPUT
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("AI INPUT DATA")
    print("=" * 70)

    print(
        input_data.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------

    try:

        prediction = model.predict(
            input_data
        )[0]

    except Exception as e:

        print("\n" + "=" * 70)
        print("PREDICTION ERROR")
        print("=" * 70)

        print("\n", e)

        return

    # ---------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------

    confidence = None

    probabilities = None

    if hasattr(model, "predict_proba"):

        try:

            probabilities = model.predict_proba(
                input_data
            )[0]

            confidence = (
                probabilities.max() * 100
            )

        except Exception:

            confidence = None

    # ---------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("             PREDICTION RESULT")
    print("=" * 70)

    print("\nLocation:")
    print(
        f"{state} "
        f"({latitude:.4f}, {longitude:.4f})"
    )

    print("\nDate:")
    print(
        f"{selected_date}"
    )

    print("\nSeason:")
    print(
        season
    )

    print("\nPredicted Landslide Trigger:")
    print(
        ">>>",
        prediction
    )

    if confidence is not None:

        print(
            f"\nAI Confidence: "
            f"{confidence:.2f}%"
        )

    # ---------------------------------------------------------
    # CLASS PROBABILITIES
    # ---------------------------------------------------------

    if probabilities is not None:

        print("\n" + "-" * 70)
        print("MODEL CLASS PROBABILITIES")
        print("-" * 70)

        try:

            classes = model.classes_

            results = sorted(
                zip(
                    classes,
                    probabilities
                ),
                key=lambda x: x[1],
                reverse=True
            )

            for cls, probability in results:

                print(
                    f"{str(cls):25} "
                    f"{probability * 100:7.2f}%"
                )

        except Exception:
            pass

    # ---------------------------------------------------------
    # INTERPRETATION
    # ---------------------------------------------------------

    trigger = str(
        prediction
    ).lower()

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    interpretations = {

        "rain":
            "Rainfall is identified as the predicted landslide trigger.",

        "continuous_rain":
            "Continuous rainfall is identified as the predicted trigger.",

        "downpour":
            "Heavy downpour is identified as the predicted trigger.",

        "flooding":
            "Flooding is identified as the predicted landslide trigger.",

        "monsoon":
            "Monsoon-related conditions are identified as the predicted trigger.",

        "construction":
            "Construction activity is identified as the predicted trigger.",

        "mining":
            "Mining activity is identified as the predicted trigger.",

        "snowfall_snowmelt":
            "Snowfall or snowmelt is identified as the predicted trigger.",

        "tropical_cyclone":
            "Tropical cyclone conditions are identified as the predicted trigger.",

        "unknown":
            "The model classified the event as an unknown trigger.",

        "other":
            "The model classified the event under another trigger category."
    }

    print(
        interpretations.get(
            trigger,
            "The model predicted: " + str(prediction)
        )
    )

    # ---------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREDICTION COMPLETED")
    print("=" * 70)

    print(
        "\nNote: This is an ML model prediction "
        "based on the supplied input data."
    )


if __name__ == "__main__":
    main()
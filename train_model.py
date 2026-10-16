import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)


DATA_FILE = "data/processed_landslides.csv"
MODEL_FILE = "model/landslide_model.pkl"
METRICS_FILE = "model/metrics.json"

TARGET = "landslide_trigger"


FEATURES = [
    "latitude",
    "longitude",
    "year",
    "month",
    "day",
    "day_of_year",
    "month_sin",
    "month_cos",
    "season",
    "state"
]


NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "year",
    "month",
    "day",
    "day_of_year",
    "month_sin",
    "month_cos"
]


CATEGORICAL_FEATURES = [
    "season",
    "state"
]


def main():

    print("=" * 70)
    print("LANDSLIDESHIELD AI - MODEL TRAINING")
    print("=" * 70)

    # -----------------------------------------------------
    # Check dataset
    # -----------------------------------------------------

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            "Processed dataset not found.\n"
            "Run:\n"
            "python preprocess_data.py"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"\nDataset rows: {len(df):,}"
    )

    # -----------------------------------------------------
    # Check features
    # -----------------------------------------------------

    missing = [
        c for c in FEATURES + [TARGET]
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    # -----------------------------------------------------
    # Clean target
    # -----------------------------------------------------

    df[TARGET] = (
        df[TARGET]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # -----------------------------------------------------
    # Remove target classes with fewer than 2 samples
    # -----------------------------------------------------

    counts = df[TARGET].value_counts()

    rare = counts[
        counts < 2
    ].index.tolist()

    if rare:

        print(
            "\nRemoving classes with fewer than "
            "2 records:"
        )

        print(rare)

        df = df[
            ~df[TARGET].isin(rare)
        ].copy()

    # -----------------------------------------------------
    # X and y
    # -----------------------------------------------------

    X = df[FEATURES].copy()

    y = df[TARGET].copy()

    if y.nunique() < 2:

        raise ValueError(
            "At least two target classes are required."
        )

    print("\nClasses:")
    print(
        y.value_counts()
        .to_string()
    )

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),

        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])

    # -----------------------------------------------------
    # Random Forest
    # -----------------------------------------------------

    classifier = RandomForestClassifier(

        n_estimators=500,

        max_depth=20,

        min_samples_split=4,

        min_samples_leaf=2,

        class_weight="balanced_subsample",

        random_state=42,

        n_jobs=-1
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            classifier
        )
    ])

    # -----------------------------------------------------
    # Train / test
    # -----------------------------------------------------

    class_counts = y.value_counts()

    stratify_value = (
        y
        if class_counts.min() >= 2
        else None
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=stratify_value
        )
    )

    print(
        f"\nTraining records: {len(X_train):,}"
    )

    print(
        f"Testing records: {len(X_test):,}"
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    print("\nTraining Random Forest...")

    pipeline.fit(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions
        )
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print(
        f"Accuracy:          {accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1:          {macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted F1:       {weighted_f1 * 100:.2f}%"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    os.makedirs(
        "model",
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_FILE
    )

    metrics = {

        "model":
            "RandomForestClassifier",

        "target":
            TARGET,

        "features":
            FEATURES,

        "classes":
            sorted(
                y.unique().tolist()
            ),

        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(balanced_accuracy),

        "macro_f1":
            float(macro_f1),

        "weighted_f1":
            float(weighted_f1),

        "training_samples":
            int(len(X_train)),

        "testing_samples":
            int(len(X_test))
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print("\n" + "=" * 70)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Model:   {MODEL_FILE}"
    )

    print(
        f"Metrics: {METRICS_FILE}"
    )


if __name__ == "__main__":
    main()
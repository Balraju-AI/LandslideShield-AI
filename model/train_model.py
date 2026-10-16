import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load cleaned data
df = pd.read_csv("data/india_landslides.csv")

# Create features
df["date"] = pd.to_datetime(df["event_date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# Features and target
X = df[["latitude", "longitude", "year", "month"]]
y = df["landslide_trigger"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Save model
joblib.dump(model, "model/landslide_model.pkl")

print("Model saved: model/landslide_model.pkl")
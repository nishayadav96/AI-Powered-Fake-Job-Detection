import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("fake_job_postings.csv", encoding="latin1")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# Select text columns
columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

# Handle missing values
for col in columns:
    data[col] = data[col].fillna("").astype(str)

# Combine job details
data["job_text"] = data[columns].agg(" ".join, axis=1)

# Convert labels into numbers
data["fraudulent"] = (
    data["fraudulent"].astype(str).str.strip().str.lower()
    .map({
        "f": 0, "false": 0, "0": 0,
        "t": 1, "true": 1, "1": 1
    })
)

data = data.dropna(subset=["fraudulent"])
data["fraudulent"] = data["fraudulent"].astype(int)

# Split data
X = data["job_text"]
y = data["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create ML model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "job_model.pkl")

print("Model trained successfully!")
print("Model saved as job_model.pkl")

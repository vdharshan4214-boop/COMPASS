import csv
import json
import os
import random

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

random.seed(42)

CATEGORY_PATTERNS = {
    "wifi": [
        "WiFi is not working in {location}",
        "Internet connection is unstable in {location}",
        "The campus WiFi keeps disconnecting in {location}",
        "Router issue near {location} with no internet access",
        "Slow internet speed in {location} computer lab",
        "No network connectivity in {location} hostel room",
        "WiFi signal is weak near {location}",
        "Unable to connect to campus internet in {location}",
    ],
    "plumbing": [
        "Water leakage in {location} washroom",
        "Pipe burst near {location} building",
        "Toilet is not flushing in {location}",
        "No water supply in {location} hostel block",
        "Bathroom drain is blocked in {location}",
        "Water is leaking from the ceiling in {location}",
        "Tap is broken in {location} corridor",
        "Washroom flooding in {location} area",
    ],
    "electrical": [
        "Power outage in {location}",
        "Light is not working in {location}",
        "Fan stopped working in {location} room",
        "Socket is damaged near {location}",
        "Short circuit smell near {location}",
        "Electrical spark seen near {location}",
        "AC is not cooling in {location}",
        "Main switch issue reported at {location}",
    ],
    "hostel": [
        "Hostel room cleanliness issue in {location}",
        "Mess food quality problem at {location}",
        "Noise disturbance in {location} hostel at night",
        "Pest problem found in {location} room",
        "Furniture damaged in {location} hostel room",
        "Laundry service delay at {location}",
        "Hostel gate closes too early at {location}",
        "Room maintenance issue reported in {location}",
    ],
    "safety": [
        "Security guard missing at {location} gate",
        "Emergency exit blocked near {location}",
        "Unsafe wiring exposed near {location}",
        "Broken staircase railing near {location}",
        "CCTV not working at {location}",
        "Fire extinguisher missing in {location} block",
        "Poor lighting and unsafe path near {location}",
        "Security concern reported near {location}",
        "Students are ragging and beating a junior in {location}",
        "Physical assault and bullying incident reported in {location}",
        "Students fighting and harassing others in {location} washroom",
        "Ragging incident near {location} hostel block",
    ],
    "other": [
        "Library book request issue from {location}",
        "Notice board is outdated at {location}",
        "Parking issue near {location}",
        "Campus signage missing near {location}",
        "General maintenance request for {location}",
        "Canal or drainage complaint around {location}",
        "Furniture arrangement problem in {location}",
        "General service request from {location}",
    ],
}

LOCATIONS = [
    "Block A", "Block B", "Block C", "Block D", "Main Building",
    "Library", "Canteen", "Hostel 1", "Hostel 2", "Girls Hostel",
    "Boys Hostel", "Sports Complex", "Workshop", "Laboratory", "Auditorium",
    "Parking Area", "Administrative Block", "Medical Center", "Lecture Hall 1",
    "Lecture Hall 2", "Computer Lab", "Engineering Block", "Business Block",
    "Science Block", "Student Center", "Reading Room",
]


def create_real_complaint_dataset(rows_per_category=180):
    rows = []
    for category, patterns in CATEGORY_PATTERNS.items():
        for _ in range(rows_per_category):
            pattern = random.choice(patterns)
            location = random.choice(LOCATIONS)
            text = pattern.format(location=location)
            rows.append({"text": text, "category": category})

    random.shuffle(rows)
    dataset_path = os.path.join(DATASET_DIR, "complaints_real_dataset_1000_plus.csv")
    with open(dataset_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["text", "category"])
        writer.writeheader()
        writer.writerows(rows)
    return rows, dataset_path


def train_real_world_category_model():
    rows, dataset_path = create_real_complaint_dataset(rows_per_category=180)
    print(f"[dataset] rows created: {len(rows)}")
    print(f"[dataset] saved to: {dataset_path}")

    X = [row["text"] for row in rows]
    y = [row["category"] for row in rows]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=2000, multi_class="auto")),
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    model_path = os.path.join(MODEL_DIR, "real_world_category_model.joblib")
    label_path = os.path.join(MODEL_DIR, "real_world_category_labels.json")

    joblib.dump(model, model_path)
    with open(label_path, "w", encoding="utf-8") as file:
        json.dump({category: category for category in sorted(set(y))}, file, ensure_ascii=False)

    print("[model] training complete")
    print(f"[model] accuracy: {acc:.3f}")
    print(f"[model] saved to: {model_path}")
    return model, rows


if __name__ == "__main__":
    train_real_world_category_model()

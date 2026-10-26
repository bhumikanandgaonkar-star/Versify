
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

DATA_PATH = Path("data/processed/versify_trial_clean.csv")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "emotion_classifier_broad.joblib"
VECTORIZER_PATH = MODEL_DIR / "emotion_classifier_vectorizer.joblib"
PREDICTIONS_PATH = MODEL_DIR / "broad_validation_predictions.csv"
CONFUSION_PATH = MODEL_DIR / "broad_emotion_confusion_matrix.png"
REPORT_PATH = MODEL_DIR / "emotion_classifier_report.csv"

LYRICS_COL = "Lyrics"
MOOD_COL = "Primary Mood"
SPLIT_COL = "Split (Train/Val/Test)"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. MAP MOODS TO BROAD CATEGORIES
# --------------------------------------------------

MOOD_MAP = {
    "Love": "Love",
    "Romantic": "Love",
    "Romantic / Mixed": "Love",
    "Tender": "Love",
    "Passionate": "Love",

    "Sad": "Sad",
    "Melancholic": "Sad",
    "Heartbreak": "Sad",
    "Heartbroken": "Sad",
    "Longing": "Sad",

    "Happy": "Happy",
    "Joyful": "Happy",
    "Celebration": "Happy",
    "Celebratory": "Happy",
    "Euphoric": "Happy",
    "Playful": "Happy",

    "Party": "Party / Energetic",
    "Energetic": "Party / Energetic",
    "Confident": "Party / Energetic",

    "Freedom / Motivation": "Motivational",
    "Motivational": "Motivational",
    "Inspiring": "Motivational",
    "Uplifting": "Motivational",
    "Hopeful": "Motivational",

    "Devotional": "Devotional / Spiritual",
    "Spiritual": "Devotional / Spiritual",
    "Party Devotional": "Devotional / Spiritual",
    "Devotional/Romantic": "Devotional / Spiritual",

    "Peaceful": "Peaceful / Reflective",
    "Thoughtful": "Peaceful / Reflective",
    "Nostalgic": "Peaceful / Reflective",
    "Dramatic": "Peaceful / Reflective",

    "Wedding": "Other",
    "Patriotic": "Other",
    "Angry": "Other",
    "Friendship": "Other",
}


# --------------------------------------------------
# 3. LOAD AND PREPARE DATASET
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

df = pd.read_csv(DATA_PATH)

required_columns = [
    LYRICS_COL,
    MOOD_COL,
    SPLIT_COL,
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df.dropna(
    subset=[LYRICS_COL, MOOD_COL, SPLIT_COL]
).copy()

df[LYRICS_COL] = df[LYRICS_COL].astype(str).str.strip()
df[MOOD_COL] = df[MOOD_COL].astype(str).str.strip()
df[SPLIT_COL] = (
    df[SPLIT_COL].astype(str).str.strip().str.lower()
)

df["Broad Mood"] = df[MOOD_COL].map(MOOD_MAP)
df = df.dropna(subset=["Broad Mood"]).copy()

train_df = df[df[SPLIT_COL] == "train"].copy()

validation_df = df[
    df[SPLIT_COL].isin(["validation", "val", "valid"])
].copy()

if train_df.empty or validation_df.empty:
    raise ValueError(
        "Training or validation data is empty. "
        "Check the dataset split column."
    )

X_train_text = train_df[LYRICS_COL]
y_train = train_df["Broad Mood"]

X_val_text = validation_df[LYRICS_COL]
y_val = validation_df["Broad Mood"]

print("Training records:", len(train_df))
print("Validation records:", len(validation_df))
print("\nTraining category counts:")
print(y_train.value_counts())


# --------------------------------------------------
# 4. TF-IDF FEATURE EXTRACTION
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=1,
    sublinear_tf=True,
    lowercase=True,
)

X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)

print("\nTraining feature shape:", X_train.shape)
print("Validation feature shape:", X_val.shape)


# --------------------------------------------------
# 5. TRAIN LINEAR SVM CLASSIFIER
# --------------------------------------------------

model = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=5000,
)

print("\nTraining Linear SVM emotion classifier...")

model.fit(X_train, y_train)

predictions = model.predict(X_val)


# --------------------------------------------------
# 6. EVALUATE MODEL
# --------------------------------------------------

accuracy = accuracy_score(y_val, predictions)

macro_f1 = f1_score(
    y_val,
    predictions,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_val,
    predictions,
    average="weighted",
    zero_division=0,
)

print("\n" + "=" * 55)
print("LINEAR SVM EMOTION CLASSIFICATION RESULTS")
print("=" * 55)

print(f"Validation accuracy: {accuracy:.4f}")
print(f"Macro F1-score:      {macro_f1:.4f}")
print(f"Weighted F1-score:   {weighted_f1:.4f}")

print("\nClassification report:")

print(
    classification_report(
        y_val,
        predictions,
        zero_division=0,
    )
)


# --------------------------------------------------
# 7. SAVE VALIDATION PREDICTIONS
# --------------------------------------------------

prediction_columns = [
    col for col in ["Song Name", LYRICS_COL, MOOD_COL]
    if col in validation_df.columns
]

prediction_df = validation_df[prediction_columns].copy()

prediction_df["Actual Broad Mood"] = y_val.values
prediction_df["Predicted Broad Mood"] = predictions

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# --------------------------------------------------
# 8. SAVE CLASSIFICATION REPORT
# --------------------------------------------------

report_dict = classification_report(
    y_val,
    predictions,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report_dict).transpose()

report_df.to_csv(REPORT_PATH)


# --------------------------------------------------
# 9. SAVE CONFUSION MATRIX
# --------------------------------------------------

labels = sorted(y_train.unique())

display = ConfusionMatrixDisplay.from_predictions(
    y_val,
    predictions,
    labels=labels,
    xticks_rotation=45,
    cmap="Blues",
    zero_division=0,
)

display.ax_.set_title("Linear SVM Emotion Confusion Matrix")

plt.tight_layout()

plt.savefig(
    CONFUSION_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# 10. SAVE TRAINED MODEL AND VECTORIZER
# --------------------------------------------------

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("\nSaved files:")
print("Model:", MODEL_PATH.resolve())
print("Vectorizer:", VECTORIZER_PATH.resolve())
print("Predictions:", PREDICTIONS_PATH.resolve())
print("Classification report:", REPORT_PATH.resolve())
print("Confusion matrix:", CONFUSION_PATH.resolve())

print("\nLinear SVM emotion classification complete.")

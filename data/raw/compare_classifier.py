
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

DATA_PATH = Path("data/processed/versify_trial_clean.csv")
OUTPUT_DIR = Path("models/classifier_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LYRICS_COL = "Lyrics"
MOOD_COL = "Primary Mood"
SPLIT_COL = "Split (Train/Val/Test)"


# --------------------------------------------------
# 2. Map original moods into eight broad categories
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
# 3. Load and prepare data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

required = [LYRICS_COL, MOOD_COL, SPLIT_COL]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(
    subset=[LYRICS_COL, MOOD_COL, SPLIT_COL]
).copy()

df[LYRICS_COL] = df[LYRICS_COL].astype(str).str.strip()
df[MOOD_COL] = df[MOOD_COL].astype(str).str.strip()
df[SPLIT_COL] = (
    df[SPLIT_COL]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["Broad Mood"] = df[MOOD_COL].map(MOOD_MAP)

unmapped = sorted(
    df.loc[df["Broad Mood"].isna(), MOOD_COL].unique()
)

if unmapped:
    raise ValueError(
        "These mood labels are missing from MOOD_MAP: "
        + ", ".join(unmapped)
    )

train_df = df[df[SPLIT_COL] == "train"].copy()

val_df = df[
    df[SPLIT_COL].isin(["validation", "val", "valid"])
].copy()

if train_df.empty or val_df.empty:
    raise ValueError(
        "Training or validation data is empty. "
        "Check split labels."
    )

X_train_text = train_df[LYRICS_COL]
X_val_text = val_df[LYRICS_COL]

y_train = train_df["Broad Mood"]
y_val = val_df["Broad Mood"]

print("Training records:", len(train_df))
print("Validation records:", len(val_df))

print("\nTraining category counts:")
print(y_train.value_counts())


# --------------------------------------------------
# 4. Create TF-IDF features using training data only
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
# 5. Define the three classifiers
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=5000,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}


# --------------------------------------------------
# 6. Train and evaluate on validation data
# --------------------------------------------------

results = []

for name, model in models.items():
    print("\n" + "=" * 60)
    print("MODEL:", name)
    print("=" * 60)

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

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

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
    })

    prediction_columns = [
        col
        for col in [
            "Song Name",
            LYRICS_COL,
            MOOD_COL,
            "Broad Mood",
        ]
        if col in val_df.columns
    ]

    prediction_df = val_df[prediction_columns].copy()

    prediction_df["Actual Broad Mood"] = y_val.to_numpy()
    prediction_df["Predicted Broad Mood"] = predictions

    safe_name = name.lower().replace(" ", "_")

    prediction_df.to_csv(
        OUTPUT_DIR / f"{safe_name}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    labels = sorted(y_train.unique())

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=labels,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    display.plot(
        xticks_rotation=45,
        values_format="d",
    )

    plt.title(f"{name} - Validation Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{safe_name}_confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # Save the trained model.
    joblib.dump(
        model,
        OUTPUT_DIR / f"{safe_name}_model.joblib",
    )


# Save the fitted TF-IDF vectorizer.
joblib.dump(
    vectorizer,
    OUTPUT_DIR / "tfidf_vectorizer.joblib",
)


# --------------------------------------------------
# 7. Compare results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False,
)

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nResults saved in:", OUTPUT_DIR)
print("Comparison complete.")


from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/raw/DOC-20261015-WA0026.csv")
OUTPUT_PATH = Path("data/processed/versify_trial_clean.csv")
CONFLICT_PATH = Path("outputs/duplicate_lyric_conflicts.csv")

LYRICS_COL = "Lyrics (Full, Section-wise)"
MOOD_COL = "Primary Mood"
SPLIT_COL = "Split (Train/Val/Test)"

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    initial_rows = len(df)

    # Remove accidental leading/trailing whitespace from text fields.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    # Normalize lyric whitespace without removing verse/chorus markers.
    df[LYRICS_COL] = (
        df[LYRICS_COL]
        .astype("string")
        .str.replace(r"[ \t]+", " ", regex=True)
        .str.strip()
    )

    # Standardize labels.
    df[MOOD_COL] = df[MOOD_COL].str.strip().str.title()

    split_map = {
        "train": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "test",
    }

    df[SPLIT_COL] = (
        df[SPLIT_COL]
        .str.strip()
        .str.lower()
        .map(split_map)
    )

    if df[SPLIT_COL].isna().any():
        bad_values = df.loc[df[SPLIT_COL].isna(), SPLIT_COL]
        raise ValueError("Unrecognized split labels found. Inspect the raw CSV.")

    # Identify lyrics associated with multiple emotion labels.
    mood_counts = df.groupby(LYRICS_COL)[MOOD_COL].nunique()
    conflicting_lyrics = mood_counts[mood_counts > 1].index

    conflicts = df[df[LYRICS_COL].isin(conflicting_lyrics)].copy()
    CONFLICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    conflicts.to_csv(CONFLICT_PATH, index=False, encoding="utf-8-sig")

    # Remove exact duplicate lyric text, retaining the first occurrence.
    # Conflicting emotion labels are separately recorded above.
    df = df.drop_duplicates(subset=[LYRICS_COL], keep="first").copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nVERSIFY CLEANING REPORT")
    print("=" * 35)
    print(f"Original rows: {initial_rows}")
    print(f"Rows after exact lyric deduplication: {len(df)}")
    print(f"Rows removed: {initial_rows - len(df)}")
    print(f"Unique lyrics remaining: {df[LYRICS_COL].nunique()}")
    print(f"Missing lyric values: {df[LYRICS_COL].isna().sum()}")
    print("\nEmotion distribution after cleaning:")
    print(df[MOOD_COL].value_counts().to_string())
    print("\nSplit distribution after cleaning:")
    print(df[SPLIT_COL].value_counts().to_string())
    print(f"\nCleaned dataset saved to: {OUTPUT_PATH}")
    print(f"Potential label conflicts saved to: {CONFLICT_PATH}")

if __name__ == "__main__":
    main()

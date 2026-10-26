
from pathlib import Path
import re
import pandas as pd

def remove_unwanted_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove Excel line-break artifacts
    text = re.sub(r"_x000d_", " ", text, flags=re.IGNORECASE)

    # Remove unwanted "Read more:" text
    text = re.sub(r"Read\s+more\s*:", " ", text, flags=re.IGNORECASE)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)

    # Remove extra spaces and blank spacing
    text = re.sub(r"\s+", " ", text).strip()

    return text


INPUT_PATH = Path("data/raw/final_dataset.csv")
OUTPUT_PATH = Path("data/processed/versify_trial_clean.csv")
CONFLICT_PATH = Path("outputs/duplicate_lyric_conflicts.csv")
DUPLICATE_PATH = Path("outputs/duplicate_lyrics_review.csv")

LYRICS_COL = "Lyrics"
MOOD_COL = "Primary Mood"
SPLIT_COL = "Split (Train/Val/Test)"

def clean_lyrics(value):
    """Remove common webpage artifacts while preserving lyric text."""
    if pd.isna(value):
        return value

    text = str(value)

    # Convert exported line-break markers into real line breaks.
    text = text.replace("_x000d_", "\n")
    text = text.replace("\\r\\n", "\n")
    text = text.replace("\\n", "\n")
    text = text.replace("\\r", "\n")

    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Remove common webpage navigation and page headings.
        if re.fullmatch(
            r"(search|follow|read more|lyrics)",
            line,
            flags=re.IGNORECASE
        ):
            continue

        if re.fullmatch(
            r"movie\s*\|\s*artist\s*\|\s*song",
            line,
            flags=re.IGNORECASE
        ):
            continue

        # Remove webpage metadata, not the actual song verses.
        if re.match(
            r"^(lyricist|composer|singer|movie|album)\s*:",
            line,
            flags=re.IGNORECASE
        ):
            continue

        # Remove lines that are clearly "Read more" links.
        if re.match(
            r"^read more\s*:",
            line,
            flags=re.IGNORECASE
        ):
            continue

        # Remove alphabet-navigation bars found on some lyric sites.
        if re.fullmatch(r"[0-9A-Z#]{10,}", line):
            continue

        # Remove a standalone webpage play control.
        if re.fullmatch(r"play", line, flags=re.IGNORECASE):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()



def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    initial_rows = len(df)

    # Verify the required columns exist before cleaning lyrics.
    required_columns = [LYRICS_COL, MOOD_COL, SPLIT_COL]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")

    # Clean common webpage artifacts from the lyrics.
    df[LYRICS_COL] = df[LYRICS_COL].map(clean_lyrics)

    # Remove remaining unwanted text from the lyrics.
    df[LYRICS_COL] = df[LYRICS_COL].map(remove_unwanted_text)


    # Remove rows where every column is empty.
    df = df.dropna(how="all").copy()
    empty_rows_removed = initial_rows - len(df)

    # Verify the required columns exist.
    required_columns = [LYRICS_COL, MOOD_COL, SPLIT_COL]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    # Clean surrounding whitespace in all text columns.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    # Normalize lyric whitespace and preserve section markers.
    df[LYRICS_COL] = (
        df[LYRICS_COL]
        .astype("string")
        .str.replace(r"[ \t]+", " ", regex=True)
        .str.strip()
        .replace("", pd.NA)
    )

    # Standardize mood labels without changing other feature columns.
    df[MOOD_COL] = (
        df[MOOD_COL]
        .astype("string")
        .str.strip()
        .str.title()
    )

    # Standardize train, validation, and test labels.
    split_map = {
        "train": "train",
        "val": "validation",
        "valid": "validation",
        "validate": "validation",
        "validation": "validation",
        "test": "test",
    }

    original_splits = (
        df[SPLIT_COL]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df[SPLIT_COL] = original_splits.map(split_map)

    # Do not guess missing or unrecognized splits for actual song rows.
    bad_splits = df[SPLIT_COL].isna()

    if bad_splits.any():
        bad_values = original_splits[bad_splits].value_counts(
            dropna=False
        ).to_dict()

        examples = df.loc[
            bad_splits,
            ["Song Name", MOOD_COL, SPLIT_COL]
        ].head(10) if "Song Name" in df.columns else df.loc[
            bad_splits, [MOOD_COL, SPLIT_COL]
        ].head(10)

        raise ValueError(
            "Some non-empty rows have missing or unrecognized split "
            f"labels: {bad_values}\n"
            "Inspect these rows before continuing.\n"
            f"{examples.to_string(index=False)}"
        )

    # Check for missing lyrics instead of silently deleting song records.
    missing_lyrics = df[LYRICS_COL].isna().sum()

    # Report repeated lyrics with different mood labels.
    valid_lyrics = df[df[LYRICS_COL].notna()].copy()

    mood_counts = valid_lyrics.groupby(LYRICS_COL)[
        MOOD_COL
    ].nunique()

    conflicting_lyrics = mood_counts[mood_counts > 1].index

    conflicts = valid_lyrics[
        valid_lyrics[LYRICS_COL].isin(conflicting_lyrics)
    ].copy()

    CONFLICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    conflicts.to_csv(
        CONFLICT_PATH, index=False, encoding="utf-8-sig"
    )

    # Report repeated lyric text for review.
    # Do not remove songs just because their lyrics match.
    duplicate_mask = valid_lyrics.duplicated(
        subset=[LYRICS_COL], keep=False
    )

    duplicates = valid_lyrics[duplicate_mask].sort_values(
        by=LYRICS_COL
    )

    DUPLICATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    duplicates.to_csv(
        DUPLICATE_PATH, index=False, encoding="utf-8-sig"
    )

    # Remove only rows identical across every column.
    before_exact_dedup = len(df)
    df = df.drop_duplicates(keep="first").copy()
    exact_rows_removed = before_exact_dedup - len(df)

    # Preserve every feature column from the input CSV.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nVERSIFY CLEANING REPORT")
    print("=" * 35)
    print(f"Original rows: {initial_rows}")
    print(f"Completely empty rows removed: {empty_rows_removed}")
    print(f"Exact duplicate rows removed: {exact_rows_removed}")
    print(f"Rows saved: {len(df)}")
    print(f"Missing lyric values: {missing_lyrics}")
    print(f"Columns preserved: {len(df.columns)}")

    print("\nMood distribution:")
    print(df[MOOD_COL].value_counts(dropna=False).to_string())

    print("\nSplit distribution:")
    print(df[SPLIT_COL].value_counts(dropna=False).to_string())

    print(f"\nCleaned dataset saved to: {OUTPUT_PATH}")
    print(f"Potential mood conflicts saved to: {CONFLICT_PATH}")
    print(f"Repeated lyrics for review saved to: {DUPLICATE_PATH}")


if __name__ == "__main__":
    main()


from pathlib import Path

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

INPUT_PATH = Path("data/processed/versify_tokenized.csv")
OUTPUT_DIR = Path("data/features/tfidf_trial")

TEXT_COL = "Lyrics (Full, Section-wise)"
MOOD_COL = "Primary Mood"
SPLIT_COL = "Split (Train/Val/Test)"

def main():
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    # Check required fields.
    required = [TEXT_COL, MOOD_COL, SPLIT_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df[SPLIT_COL] = df[SPLIT_COL].str.strip().str.lower()

    # Use only known split names.
    split_map = {
        "train": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "test",
    }
    df[SPLIT_COL] = df[SPLIT_COL].map(split_map)

    if df[SPLIT_COL].isna().any():
        raise ValueError("Unknown split label found.")

    train = df[df[SPLIT_COL] == "train"]
    validation = df[df[SPLIT_COL] == "validation"]
    test = df[df[SPLIT_COL] == "test"]

    if train.empty or validation.empty or test.empty:
        raise ValueError("Train, validation, and test splits must all be non-empty.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Word-level TF-IDF: unigrams and bigrams.
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=10000,
        sublinear_tf=True,
    )

    X_train_word = word_vectorizer.fit_transform(train[TEXT_COL])
    X_val_word = word_vectorizer.transform(validation[TEXT_COL])
    X_test_word = word_vectorizer.transform(test[TEXT_COL])

    save_npz(OUTPUT_DIR / "word_train.npz", X_train_word)
    save_npz(OUTPUT_DIR / "word_validation.npz", X_val_word)
    save_npz(OUTPUT_DIR / "word_test.npz", X_test_word)
    joblib.dump(word_vectorizer, OUTPUT_DIR / "word_vectorizer.joblib")

    # Character-level TF-IDF.
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
        max_features=10000,
        sublinear_tf=True,
    )

    X_train_char = char_vectorizer.fit_transform(train[TEXT_COL])
    X_val_char = char_vectorizer.transform(validation[TEXT_COL])
    X_test_char = char_vectorizer.transform(test[TEXT_COL])

    save_npz(OUTPUT_DIR / "char_train.npz", X_train_char)
    save_npz(OUTPUT_DIR / "char_validation.npz", X_val_char)
    save_npz(OUTPUT_DIR / "char_test.npz", X_test_char)
    joblib.dump(char_vectorizer, OUTPUT_DIR / "char_vectorizer.joblib")

    # Keep labels and record IDs aligned with each matrix's row order.
    for split_name, split_df in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        split_df[["ID", MOOD_COL, SPLIT_COL]].to_csv(
            OUTPUT_DIR / f"{split_name}_labels.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\nVERSIFY FEATURE EXTRACTION REPORT")
    print("=" * 40)

    print("\nWord-level TF-IDF:")
    print("Train matrix:", X_train_word.shape)
    print("Validation matrix:", X_val_word.shape)
    print("Test matrix:", X_test_word.shape)

    print("\nCharacter-level TF-IDF:")
    print("Train matrix:", X_train_char.shape)
    print("Validation matrix:", X_val_char.shape)
    print("Test matrix:", X_test_char.shape)

    print("\nOutput directory:", OUTPUT_DIR)

if __name__ == "__main__":
    main()


from pathlib import Path
import re
import json
import pandas as pd
from collections import Counter

INPUT_PATH = Path("data/processed/versify_trial_clean.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_CSV = OUTPUT_DIR / "versify_tokenized.csv"
VOCAB_JSON = OUTPUT_DIR / "versify_vocab.json"

LYRICS_COL = "Lyrics (Full, Section-wise)"
MOOD_COL = "Primary Mood"

# Keeps section markers, English/Hindi words, and numbers as tokens.
TOKEN_PATTERN = re.compile(
    r"\[[^\]]+\]|[\w]+(?:['’-][\w]+)*|[^\w\s]",
    flags=re.UNICODE
)

def tokenize(text):
    if pd.isna(text):
        return []
    return TOKEN_PATTERN.findall(str(text).lower())

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    df["Tokens"] = df[LYRICS_COL].apply(tokenize)
    df["Token Count"] = df["Tokens"].apply(len)

    # Build vocabulary from the training split only to avoid data leakage.
    train_mask = df["Split (Train/Val/Test)"].eq("train")
    token_counter = Counter(
        token
        for tokens in df.loc[train_mask, "Tokens"]
        for token in tokens
    )

    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    vocabulary = {
        token: index
        for index, token in enumerate(special_tokens)
    }

    for token, _ in token_counter.most_common():
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    # Convert tokens to integer IDs. Unseen validation/test tokens become <UNK>.
    df["Token IDs"] = df["Tokens"].apply(
        lambda tokens: [
            vocabulary.get(token, vocabulary["<UNK>"])
            for token in tokens
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON strings make list columns portable in CSV/Excel.
    df["Tokens"] = df["Tokens"].apply(
        lambda values: json.dumps(values, ensure_ascii=False)
    )
    df["Token IDs"] = df["Token IDs"].apply(json.dumps)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    VOCAB_JSON.write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nVERSIFY TOKENIZATION REPORT")
    print("=" * 35)
    print(f"Records processed: {len(df)}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Training records: {train_mask.sum()}")
    print(f"Average tokens per lyric: {df['Token Count'].mean():.2f}")
    print(f"Maximum tokens in one lyric: {df['Token Count'].max()}")

    print("\nExample:")
    example = df.iloc[0]
    print("Mood:", example[MOOD_COL])
    print("Tokens:", example["Tokens"])
    print("Token IDs:", example["Token IDs"])

    print(f"\nTokenized CSV saved to: {OUTPUT_CSV}")
    print(f"Vocabulary saved to: {VOCAB_JSON}")

if __name__ == "__main__":
    main()

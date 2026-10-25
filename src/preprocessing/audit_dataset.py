
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/raw/DOC-20261015-WA0026.csv")
OUTPUT_PATH = Path("outputs/dataset_audit.txt")

df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

lyrics_col = "Lyrics (Full, Section-wise)"
mood_col = "Primary Mood"
split_col = "Split (Train/Val/Test)"

report = []

report.append("VERSIFY DATASET AUDIT")
report.append("=" * 40)
report.append(f"Rows: {len(df)}")
report.append(f"Columns: {len(df.columns)}")
report.append(
    f"Exact duplicate lyric rows: "
    f"{df[lyrics_col].fillna('').str.strip().duplicated().sum()}"
)
report.append(
    f"Unique lyric texts: "
    f"{df[lyrics_col].fillna('').str.strip().nunique()}"
)
report.append("\nMissing values:")
report.append(df.isnull().sum().to_string())

report.append("\n\nEmotion distribution:")
report.append(df[mood_col].value_counts(dropna=False).to_string())

report.append("\n\nOriginal split distribution:")
report.append(df[split_col].value_counts(dropna=False).to_string())

# Standardize split names without changing the original file.
df[split_col] = (
    df[split_col]
    .astype("string")
    .str.strip()
    .str.lower()
    .replace({"val": "validation", "valid": "validation"})
)

report.append("\n\nStandardized split distribution:")
report.append(df[split_col].value_counts(dropna=False).to_string())

report_text = "\n".join(report)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(report_text, encoding="utf-8")

print(report_text)
print(f"\nAudit report saved to: {OUTPUT_PATH}")

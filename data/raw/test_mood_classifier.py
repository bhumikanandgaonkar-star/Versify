
from pathlib import Path
import joblib

MODEL_PATH = Path("models/emotion_classifier_broad.joblib")
VECTORIZER_PATH = Path("models/emotion_classifier_vectorizer.joblib")

if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    raise FileNotFoundError(
        "Model or vectorizer not found. Run this script "
        "from the Versify project root."
    )

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("VERSIFY MOOD CLASSIFIER TEST")
print("Type a lyric or a short text.")
print("Type 'exit' to stop.")

while True:
    text = input("\nEnter text: ").strip()

    if text.lower() == "exit":
        print("Testing completed.")
        break

    if not text:
        print("Please enter some text.")
        continue

    features = vectorizer.transform([text])
    predicted_mood = model.predict(features)[0]

    print("Predicted mood:", predicted_mood)

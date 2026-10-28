
import os
import time
import numpy as np
import pandas as pd
import soundfile as sf
import laion_clap

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_ROOT, "meta", "esc50.csv")
AUDIO_DIR = os.path.join(DATA_ROOT, "audio")
OUTPUT_DIR = os.path.join(DATA_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ["dog", "rain", "car_horn", "chainsaw", "clock_tick"]

TEXT_LABELS = {
    "dog": "The sound of a dog barking.",
    "rain": "The sound of rain falling.",
    "car_horn": "The sound of a car horn honking.",
    "chainsaw": "The sound of a chainsaw running.",
    "clock_tick": "The sound of a clock ticking."
}

# --------------------------------------------------
# Load the same five classes and fold 5 test set
# --------------------------------------------------
df = pd.read_csv(CSV_PATH)
test_df = df[
    (df["category"].isin(CLASSES)) &
    (df["fold"] == 5)
].copy().reset_index(drop=True)

if len(test_df) != 40:
    raise ValueError(
        f"Expected 40 test clips, found {len(test_df)}. "
        "Check the dataset path and fold selection."
    )

audio_paths = [
    os.path.join(AUDIO_DIR, filename)
    for filename in test_df["filename"]
]

for path in audio_paths:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing audio file: {path}")

print(f"Test clips: {len(test_df)}")
print(f"Classes: {CLASSES}")

# --------------------------------------------------
# Load pretrained CLAP
# --------------------------------------------------
print("\nLoading pretrained CLAP model...")
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

# --------------------------------------------------
# Compute audio and text embeddings
# --------------------------------------------------
print("\nExtracting audio embeddings...")
start = time.perf_counter()

audio_embeddings = model.get_audio_embedding_from_filelist(
    x=audio_paths,
    use_tensor=False
)

audio_time = time.perf_counter() - start

print("Extracting text embeddings...")
text_prompts = [TEXT_LABELS[c] for c in CLASSES]
text_embeddings = model.get_text_embedding(
    text_prompts,
    use_tensor=False
)

# --------------------------------------------------
# Compare embeddings using cosine similarity
# --------------------------------------------------
audio_embeddings = np.asarray(audio_embeddings, dtype=np.float32)
text_embeddings = np.asarray(text_embeddings, dtype=np.float32)

audio_embeddings /= np.maximum(
    np.linalg.norm(audio_embeddings, axis=1, keepdims=True), 1e-12
)
text_embeddings /= np.maximum(
    np.linalg.norm(text_embeddings, axis=1, keepdims=True), 1e-12
)

similarities = audio_embeddings @ text_embeddings.T
predicted_indices = np.argmax(similarities, axis=1)
predictions = [CLASSES[i] for i in predicted_indices]
actual = test_df["category"].tolist()

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
accuracy = accuracy_score(actual, predictions)
macro_f1 = f1_score(
    actual, predictions, labels=CLASSES,
    average="macro", zero_division=0
)
report = classification_report(
    actual, predictions, labels=CLASSES,
    zero_division=0, output_dict=True
)

test_df["predicted_category"] = predictions
test_df["correct"] = (
    test_df["category"] == test_df["predicted_category"]
)
test_df["inference_time_seconds"] = audio_time / len(test_df)

for i, category in enumerate(CLASSES):
    test_df[f"similarity_{category}"] = similarities[:, i]

test_df.to_csv(
    os.path.join(OUTPUT_DIR, "clap_test_predictions.csv"),
    index=False
)

metrics = pd.DataFrame([{
    "model": "CLAP zero-shot",
    "test_clips": len(test_df),
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "audio_embedding_time_seconds": audio_time,
    "average_audio_embedding_time_per_clip_seconds":
        audio_time / len(test_df)
}])

metrics.to_csv(
    os.path.join(OUTPUT_DIR, "clap_metrics.csv"),
    index=False
)

pd.DataFrame(
    confusion_matrix(actual, predictions, labels=CLASSES),
    index=CLASSES,
    columns=CLASSES
).to_csv(os.path.join(OUTPUT_DIR, "clap_confusion_matrix.csv"))

pd.DataFrame(report).T.to_csv(
    os.path.join(OUTPUT_DIR, "clap_classification_report.csv")
)

print("\n========== CLAP RESULTS ==========")
print(f"Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)")
print(f"Macro-F1: {macro_f1:.3f}")
print(f"Audio embedding time: {audio_time:.2f} seconds")
print(f"Average audio embedding time: "
      f"{audio_time / len(test_df):.3f} seconds/clip")
print("\nClassification report:")
print(classification_report(actual, predictions, labels=CLASSES,
                            zero_division=0))
print("\nConfusion matrix (rows=actual, columns=predicted):")
print(pd.DataFrame(
    confusion_matrix(actual, predictions, labels=CLASSES),
    index=CLASSES, columns=CLASSES
))
print(f"\nResults saved in: {OUTPUT_DIR}")

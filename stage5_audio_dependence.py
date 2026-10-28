
import os
import numpy as np
import pandas as pd
import laion_clap
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT, "meta", "esc50.csv")
AUDIO_DIR = os.path.join(ROOT, "audio")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = ["dog", "rain", "car_horn", "chainsaw", "clock_tick"]

PROMPTS = [
    "The sound of a dog barking.",
    "The sound of rain falling.",
    "The sound of a car horn honking.",
    "The sound of a chainsaw running.",
    "The sound of a clock ticking."
]

df = pd.read_csv(CSV_PATH)
test = df[
    (df["fold"] == 5) & df["category"].isin(CLASSES)
].reset_index(drop=True)

if len(test) != 40:
    raise ValueError(f"Expected 40 clips, found {len(test)}")

paths = [os.path.join(AUDIO_DIR, f) for f in test["filename"]]
actual = test["category"].tolist()

# Create a random shuffle with no clip remaining in its original position.
rng = np.random.default_rng(42)
original_indices = np.arange(len(paths))

while True:
    permutation = rng.permutation(len(paths))
    if np.all(permutation != original_indices):
        break

shuffled_paths = [paths[i] for i in permutation]

print("Loading CLAP...")
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

print("Extracting text embeddings...")
text_emb = np.asarray(
    model.get_text_embedding(PROMPTS, use_tensor=False),
    dtype=np.float32
)
text_emb /= np.maximum(
    np.linalg.norm(text_emb, axis=1, keepdims=True), 1e-12
)

results = []

for condition, audio_paths in [
    ("correct_audio", paths),
    ("shuffled_audio", shuffled_paths)
]:
    print(f"Evaluating {condition}...")

    audio_emb = np.asarray(
        model.get_audio_embedding_from_filelist(
            x=audio_paths, use_tensor=False
        ),
        dtype=np.float32
    )
    audio_emb /= np.maximum(
        np.linalg.norm(audio_emb, axis=1, keepdims=True), 1e-12
    )

    scores = audio_emb @ text_emb.T
    predictions = [CLASSES[i] for i in np.argmax(scores, axis=1)]

    for i in range(len(test)):
        results.append({
            "filename": test.loc[i, "filename"],
            "actual_category": actual[i],
            "condition": condition,
            "audio_filename_used": os.path.basename(audio_paths[i]),
            "predicted_category": predictions[i],
            "correct": predictions[i] == actual[i]
        })

result_df = pd.DataFrame(results)
result_df.to_csv(
    os.path.join(OUT_DIR, "clap_audio_dependence_predictions.csv"),
    index=False
)

summary_rows = []
for condition, group in result_df.groupby("condition", sort=False):
    summary_rows.append({
        "condition": condition,
        "test_clips": len(group),
        "accuracy": accuracy_score(
            group["actual_category"], group["predicted_category"]
        ),
        "macro_f1": f1_score(
            group["actual_category"],
            group["predicted_category"],
            labels=CLASSES,
            average="macro",
            zero_division=0
        )
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(
    os.path.join(OUT_DIR, "clap_audio_dependence_summary.csv"),
    index=False
)

print("\n========== AUDIO DEPENDENCE RESULTS ==========")
print(summary.to_string(index=False))
print("\nSaved results to outputs/")


import os
import time
import numpy as np
import pandas as pd
import soundfile as sf
import laion_clap

# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------
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

# --------------------------------------------------
# Load exactly the same 40 test clips
# --------------------------------------------------
df = pd.read_csv(CSV_PATH)
test = df[(df["fold"] == 5) & (df["category"].isin(CLASSES))]
test = test.reset_index(drop=True)

if len(test) != 40:
    raise ValueError(f"Expected 40 test clips, found {len(test)}")

paths = [os.path.join(AUDIO_DIR, f) for f in test["filename"]]

# --------------------------------------------------
# Add noise at a specified SNR
# SNR = 10*log10(signal power / noise power)
# --------------------------------------------------
def add_noise_at_snr(audio, snr_db, rng):
    audio = audio.astype(np.float32)
    signal_power = np.mean(audio ** 2)

    if signal_power < 1e-12:
        return audio.copy()

    noise = rng.normal(0, 1, size=audio.shape).astype(np.float32)
    noise_power = np.mean(noise ** 2)
    target_noise_power = signal_power / (10 ** (snr_db / 10))

    noise *= np.sqrt(target_noise_power / max(noise_power, 1e-12))
    return audio + noise

# --------------------------------------------------
# Load pretrained CLAP
# --------------------------------------------------
print("Loading pretrained CLAP...")
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

print("Extracting text embeddings...")
text_embeddings = np.asarray(
    model.get_text_embedding(PROMPTS, use_tensor=False),
    dtype=np.float32
)
text_embeddings /= np.maximum(
    np.linalg.norm(text_embeddings, axis=1, keepdims=True), 1e-12
)

# --------------------------------------------------
# Prepare clean and noisy copies
# Temporary WAV files are stored in outputs/noisy_audio
# --------------------------------------------------
temp_dir = os.path.join(OUT_DIR, "noisy_audio")
os.makedirs(temp_dir, exist_ok=True)

rng = np.random.default_rng(42)
conditions = {"clean": None, "snr_10dB": 10, "snr_5dB": 5}
records = []

for condition, snr in conditions.items():
    print(f"\nEvaluating condition: {condition}")
    condition_paths = []

    if condition == "clean":
        condition_paths = paths
    else:
        for i, path in enumerate(paths):
            audio, sr = sf.read(path, dtype="float32")

            # ESC-50 clips are mono, but handle stereo defensively.
            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            noisy = add_noise_at_snr(audio, snr, rng)
            noisy_path = os.path.join(
                temp_dir, f"{condition}_{i:02d}.wav"
            )
            sf.write(noisy_path, noisy, sr)
            condition_paths.append(noisy_path)

    start = time.perf_counter()
    audio_embeddings = np.asarray(
        model.get_audio_embedding_from_filelist(
            x=condition_paths, use_tensor=False
        ),
        dtype=np.float32
    )
    elapsed = time.perf_counter() - start

    audio_embeddings /= np.maximum(
        np.linalg.norm(audio_embeddings, axis=1, keepdims=True), 1e-12
    )
    similarities = audio_embeddings @ text_embeddings.T
    predicted_indices = np.argmax(similarities, axis=1)
    predictions = [CLASSES[i] for i in predicted_indices]

    for i, (_, row) in enumerate(test.iterrows()):
        records.append({
            "filename": row["filename"],
            "actual_category": row["category"],
            "condition": condition,
            "snr_db": snr if snr is not None else "clean",
            "predicted_category": predictions[i],
            "correct": predictions[i] == row["category"],
            "max_similarity": float(similarities[i].max()),
            "embedding_time_seconds": elapsed,
            "average_time_per_clip_seconds": elapsed / len(test)
        })

# --------------------------------------------------
# Save results
# --------------------------------------------------
pred_df = pd.DataFrame(records)
pred_df.to_csv(
    os.path.join(OUT_DIR, "clap_noise_predictions.csv"), index=False
)

summary = pred_df.groupby("condition", sort=False).agg(
    test_clips=("correct", "size"),
    correct_predictions=("correct", "sum"),
    accuracy=("correct", "mean"),
    mean_similarity=("max_similarity", "mean"),
    embedding_time_seconds=("embedding_time_seconds", "first"),
    average_time_per_clip_seconds=("average_time_per_clip_seconds", "first")
).reset_index()

summary["accuracy_percent"] = summary["accuracy"] * 100
summary.to_csv(
    os.path.join(OUT_DIR, "clap_noise_summary.csv"), index=False
)

print("\n========== NOISE ROBUSTNESS RESULTS ==========")
print(summary.to_string(index=False))
print("\nSaved:")
print(os.path.join(OUT_DIR, "clap_noise_predictions.csv"))
print(os.path.join(OUT_DIR, "clap_noise_summary.csv"))

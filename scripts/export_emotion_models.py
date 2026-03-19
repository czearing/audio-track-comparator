#!/usr/bin/env python3
"""
Download the Essentia Discogs-EffNet emotion classification-head models (ONNX).

These models take 1280-dimensional embeddings from the Discogs-EffNet backbone
and output a 2-class softmax.

  input:  model/Placeholder   [1280]  float32   (EffNet embedding, one patch)
  output: model/Softmax       [2]    float32   (class probabilities)

Index 0 of the softmax is the positive class probability:
  mood_happy-discogs-effnet-1      → index 0 = P(happy)      → mood_dark_to_happy
  mood_aggressive-discogs-effnet-1 → index 0 = P(aggressive) → mood_aggressive

Files saved to ~/.cache/audio-track-comparator/emotion/:
  valence.onnx    (mood_happy-discogs-effnet-1)
  arousal.onnx    (mood_aggressive-discogs-effnet-1)
"""

import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://essentia.upf.edu/models/classification-heads"

MODELS = {
    "valence.onnx": (
        f"{BASE_URL}/mood_happy/mood_happy-discogs-effnet-1.onnx"
    ),
    "arousal.onnx": (
        f"{BASE_URL}/mood_aggressive/mood_aggressive-discogs-effnet-1.onnx"
    ),
}


def download_with_progress(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    print(f"  -> {dest}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1_048_576
            total_mb = total_size / 1_048_576
            print(f"\r  {pct}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()  # newline after progress


def main() -> None:
    cache_dir = Path.home() / ".cache" / "audio-track-comparator" / "emotion"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in MODELS.items():
        dest = cache_dir / filename
        if not dest.exists():
            download_with_progress(url, dest)
        else:
            size_kb = dest.stat().st_size / 1024
            print(f"Already present ({size_kb:.0f} KB): {dest}")

        # Sanity check
        size_kb = dest.stat().st_size / 1024
        if size_kb < 50:
            print(f"ERROR: {filename} seems too small ({size_kb:.0f} KB). Download may have failed.")
            sys.exit(1)

    print()
    print("Emotion models ready.")
    for filename in MODELS:
        dest = cache_dir / filename
        size_kb = dest.stat().st_size / 1024
        print(f"  {filename}: {dest}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()

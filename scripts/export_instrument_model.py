#!/usr/bin/env python3
"""
Download the Essentia MTT MusiCNN auto-tagging model (ONNX) and its label list.

The model predicts 50 MagnaTagATune tags that cover instruments, genre, and mood.
We use it purely as an instrument detector by filtering to instrument-relevant tags
at inference time in the Rust code.

Model details:
  input:  [187, 96]  or  [n_patches, 187, 96]  float32  (mel spectrogram patches)
  output: [1, 50]    or  [n_patches, 50]        float32  (sigmoid tag probabilities)

Preprocessing (matches TensorflowPredictMusiCNN / TensorflowInputMusiCNN):
  sample_rate = 16000
  frame_size  = 512    (FFT window)
  hop_size    = 256
  n_mels      = 96
  patch_size  = 187    (frames per model input patch)
  patch_hop   = 93     (frames between patch starts, 50% overlap)

Files saved to ~/.cache/audio-track-comparator/instruments/:
  instrument_detector.onnx   — the ONNX model
  instrument_labels.json     — ordered list of 50 tag strings
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

ONNX_URL = (
    "https://essentia.upf.edu/models/autotagging/mtt/mtt-musicnn-1.onnx"
)

# All 50 MagnaTagATune labels in model output order (index 0..49).
# Sourced from mtt-musicnn-1.json at essentia.upf.edu.
LABELS = [
    "guitar",
    "classical",
    "slow",
    "techno",
    "strings",
    "drums",
    "electronic",
    "rock",
    "fast",
    "piano",
    "ambient",
    "beat",
    "violin",
    "vocal",
    "synth",
    "female",
    "indian",
    "opera",
    "male",
    "singing",
    "vocals",
    "no vocals",
    "harpsichord",
    "loud",
    "quiet",
    "flute",
    "woman",
    "male vocal",
    "no vocal",
    "pop",
    "soft",
    "sitar",
    "solo",
    "man",
    "classic",
    "choir",
    "voice",
    "new age",
    "dance",
    "male voice",
    "female vocal",
    "beats",
    "harp",
    "cello",
    "no voice",
    "weird",
    "country",
    "metal",
    "female voice",
    "choral",
]

assert len(LABELS) == 50, f"Expected 50 labels, got {len(LABELS)}"


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
    cache_dir = Path.home() / ".cache" / "audio-track-comparator" / "instruments"
    cache_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = cache_dir / "instrument_detector.onnx"
    labels_path = cache_dir / "instrument_labels.json"

    # Save labels
    if not labels_path.exists():
        print(f"Writing instrument labels -> {labels_path}")
        labels_path.write_text(json.dumps(LABELS, indent=2, ensure_ascii=False))
    else:
        print(f"Labels already present: {labels_path}")

    # Download ONNX model
    if not onnx_path.exists():
        download_with_progress(ONNX_URL, onnx_path)
    else:
        size_mb = onnx_path.stat().st_size / 1_048_576
        print(f"ONNX model already present ({size_mb:.1f} MB): {onnx_path}")

    # Basic sanity check
    size_mb = onnx_path.stat().st_size / 1_048_576
    if size_mb < 1:
        print(f"ERROR: ONNX file seems too small ({size_mb:.1f} MB). Download may have failed.")
        sys.exit(1)

    print()
    print("Instrument model ready.")
    print(f"  ONNX:   {onnx_path}")
    print(f"  Labels: {labels_path}")


if __name__ == "__main__":
    main()

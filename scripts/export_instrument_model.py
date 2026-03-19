#!/usr/bin/env python3
"""
Download the Essentia MTG-Jamendo instrument classifier (Discogs-EffNet head).

This model takes 1280-dimensional embeddings from the Discogs-EffNet backbone
and outputs multi-label sigmoid probabilities over ~40 instrument categories.

  input:  model/Placeholder  [1280]  float32   (EffNet embedding)
  output: model/Sigmoid       [N]    float32   (per-instrument probability, 0-1)

Uses the same Discogs-EffNet backbone already run for genre/quality — no extra
preprocessing or resampling required.

Files saved to ~/.cache/audio-track-comparator/instruments/:
  instrument_detector.onnx   — the ONNX model
  instrument_labels.json     — ordered list of instrument class names
"""

import json
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument"
ONNX_URL = f"{BASE_URL}/mtg_jamendo_instrument-discogs-effnet-1.onnx"
META_URL = f"{BASE_URL}/mtg_jamendo_instrument-discogs-effnet-1.json"


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

    # Download ONNX model
    if not onnx_path.exists():
        download_with_progress(ONNX_URL, onnx_path)
    else:
        size_kb = onnx_path.stat().st_size / 1024
        print(f"Already present ({size_kb:.0f} KB): {onnx_path}")

    size_kb = onnx_path.stat().st_size / 1024
    if size_kb < 50:
        print(f"ERROR: ONNX file too small ({size_kb:.0f} KB). Download may have failed.")
        sys.exit(1)

    # Download metadata JSON and extract class labels
    if not labels_path.exists():
        meta_dest = cache_dir / "_meta.json"
        download_with_progress(META_URL, meta_dest)
        meta = json.loads(meta_dest.read_text())
        classes = meta.get("classes", meta.get("tags", []))
        if not classes:
            print("ERROR: Could not find 'classes' key in metadata JSON.")
            print(f"Keys present: {list(meta.keys())}")
            sys.exit(1)
        labels_path.write_text(json.dumps(classes, indent=2, ensure_ascii=False))
        meta_dest.unlink()
        print(f"Wrote {len(classes)} instrument labels -> {labels_path}")
    else:
        classes = json.loads(labels_path.read_text())
        print(f"Labels already present ({len(classes)} classes): {labels_path}")

    print()
    print("Instrument model ready.")
    print(f"  ONNX:   {onnx_path}  ({onnx_path.stat().st_size / 1024:.0f} KB)")
    print(f"  Labels: {labels_path}  ({len(classes)} classes)")


if __name__ == "__main__":
    main()

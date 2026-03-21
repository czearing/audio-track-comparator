#!/usr/bin/env python3
"""
Download the Essentia Discogs-EffNet multi-task contrastive embedding model.

This model was trained with a contrastive objective targeting artist and track
associations in a multi-task setup — purpose-built for music similarity, NOT
audio event classification. Cosine similarity between two tracks' mean-pooled
embeddings gives a perceptually meaningful music similarity score.

  input:  [N, 128, 96]  float32  (N mel patches, 128 frames, 96 mel bins)
  output: [N, 1280]     float32  (per-patch contrastive embeddings)

Preprocessing is identical to the Discogs-EffNet classification backbone:
  sample_rate = 16 000 Hz
  frame_size  = 512 samples
  hop_size    = 256 samples
  n_mels      = 96
  patch_size  = 128 frames

File saved to ~/.cache/audio-track-comparator/similarity/similarity.onnx
"""

import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/"
    "discogs_multi_embeddings-effnet-bs64-1.onnx"
)


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
    print()


def main() -> None:
    cache_dir = Path.home() / ".cache" / "audio-track-comparator" / "similarity"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dest = cache_dir / "similarity.onnx"

    if not dest.exists():
        download_with_progress(MODEL_URL, dest)
    else:
        size_mb = dest.stat().st_size / 1_048_576
        print(f"Already present ({size_mb:.1f} MB): {dest}")

    size_mb = dest.stat().st_size / 1_048_576
    if size_mb < 5:
        print(f"ERROR: Model too small ({size_mb:.1f} MB). Download may have failed.")
        sys.exit(1)

    # Sanity check: verify input/output shapes via onnxruntime
    try:
        import numpy as np
        import onnxruntime as rt

        sess = rt.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
        dummy = np.zeros((64, 128, 96), dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: dummy})
        assert out[0].shape == (64, 1280), (
            f"Expected output shape (64, 1280), got {out[0].shape}"
        )
        print(f"Sanity check passed: input [64, 128, 96] -> output {out[0].shape}")
    except ImportError:
        print("onnxruntime not installed — skipping sanity check.")
    except Exception as e:
        print(f"ERROR: Sanity check failed: {e}")
        sys.exit(1)

    print()
    print("Similarity model ready.")
    print(f"  {dest}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

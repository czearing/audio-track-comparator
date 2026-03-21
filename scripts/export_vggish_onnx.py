"""
One-time export script for the VGGish audio embedding model.

Exports:
  vggish.onnx
    Input:  patches  [N, 1, 96, 64]  float32  (N log-mel patches, 1 channel, 96 time frames, 64 mel bins)
    Output: embeddings [N, 128]   float32  (one 128-dim embedding per patch)

Cache directory: ~/.cache/audio-track-comparator/vggish/

Requirements:
    pip install torchvggish

Usage:
    python scripts/export_vggish_onnx.py
"""

import sys

# Ensure UTF-8 output (avoids UnicodeEncodeError on Windows with non-ASCII log symbols)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path


def get_cache_dir() -> Path:
    cache_dir = Path.home() / ".cache" / "audio-track-comparator" / "vggish"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def export_vggish(cache_dir: Path) -> Path:
    import torch
    from torchvggish import vggish

    onnx_path = cache_dir / "vggish.onnx"
    if onnx_path.exists():
        print(f"VGGish ONNX already exists: {onnx_path}")
        return onnx_path

    print("Loading VGGish model...")
    model = vggish()
    model.eval()
    print("Model loaded.")
    print()

    # VGGish takes [N, 1, 96, 64] log-mel patches (with channel dim) and returns [N, 128] embeddings.
    dummy = torch.zeros(2, 1, 96, 64)

    # Forward-pass sanity check — must pass before writing the ONNX file.
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 128), f"Unexpected VGGish output shape: {out.shape}"
    print(f"  Forward pass OK: input shape {tuple(dummy.shape)}, output shape {tuple(out.shape)}")

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["patches"],
        output_names=["embeddings"],
        dynamic_axes={"patches": {0: "N"}, "embeddings": {0: "N"}},
        opset_version=14,
        do_constant_folding=True,
    )

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"  VGGish exported: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


def main():
    try:
        import torch  # noqa: F401
        import torchvggish  # noqa: F401
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}", file=sys.stderr)
        print("Install with: pip install torchvggish", file=sys.stderr)
        sys.exit(1)

    cache_dir = get_cache_dir()
    print(f"Cache directory: {cache_dir}")
    print()

    onnx_path = export_vggish(cache_dir)
    print()
    print("=" * 50)
    print("Export complete.")
    print(f"  VGGish model: {onnx_path.stat().st_size / 1e6:.2f} MB")
    print(f"  Path: {onnx_path}")


if __name__ == "__main__":
    main()

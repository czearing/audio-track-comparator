"""
One-time export script for the CLAP audio encoder.

Exports:
  1. clap-htsat-unfused-audio.onnx  -- audio encoder
     Input:  mel_spectrogram [batch, 1, 1001, 64] f32  (time x mels)
     Output: audio_embedding  [batch, 512] f32
  2. clap-htsat-unfused-text.onnx   -- text encoder (optional)
  3. text_embeddings.bin            -- pre-computed, L2-normalized text embeddings [157, 512]

Cache directory: ~/.cache/audio-track-comparator/clap/

Requirements:
    pip install torch transformers numpy

Usage:
    python scripts/export_clap_onnx.py
"""

import os
import sys
import warnings

# Ensure UTF-8 output (avoids UnicodeEncodeError on Windows with non-ASCII log symbols)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

from pathlib import Path

# ---------------------------------------------------------------------------
# Vocabulary (must match src/vocab.rs exactly — same order)
# ---------------------------------------------------------------------------
INSTRUMENTS = [
    "acoustic guitar", "electric guitar", "bass guitar", "distorted guitar",
    "piano", "electric piano", "synthesizer", "synth lead", "synth pad",
    "drums", "drum machine", "hi-hat", "kick drum", "snare",
    "violin", "cello", "strings", "brass", "trumpet", "saxophone",
    "flute", "choir", "vocals", "rap vocals", "808 bass",
    "sub bass", "organ", "harmonica", "banjo", "ukulele",
]
MOOD = [
    "dark", "bright", "melancholic", "euphoric", "aggressive", "chill",
    "romantic", "haunting", "uplifting", "mysterious", "nostalgic",
    "tense", "playful", "dreamy", "raw", "emotional", "epic",
]
ENERGY = ["low energy", "medium energy", "high energy", "intense"]
GENRE = [
    # EDM broad
    "edm", "electronic", "dance music",
    # House
    "house", "deep house", "tech house", "progressive house", "electro house",
    "big room house", "future house", "bass house", "afro house",
    # Techno
    "techno", "industrial techno", "minimal techno",
    # Trance
    "trance", "psytrance", "progressive trance",
    # Dubstep / bass
    "dubstep", "brostep", "melodic dubstep", "riddim", "future bass", "wave",
    # Drum and bass
    "drum and bass", "liquid drum and bass", "neurofunk", "jump up", "halftime",
    # Trap / hip hop
    "trap", "dark trap", "melodic trap", "phonk", "hip hop", "boom bap",
    "drill", "grime", "cloud rap",
    # Lo-fi
    "lo-fi", "lo-fi hip hop",
    # Hardstyle / hardcore
    "hardstyle", "hardcore", "gabber",
    # Breakbeat / jungle
    "breakbeat", "jungle", "uk garage",
    # Pop
    "pop", "dance pop", "electropop", "hyperpop", "indie pop", "synthpop",
    "dark pop", "k-pop",
    # Rock
    "rock", "alternative", "indie rock", "post-rock", "shoegaze", "grunge",
    "punk", "pop punk", "emo",
    # Metal
    "metal", "heavy metal", "metalcore", "nu-metal", "death metal", "black metal",
    # Ambient / experimental
    "ambient", "idm", "glitch", "experimental",
    # Synthwave / retro
    "synthwave", "retrowave", "vaporwave", "chillwave",
    # R&B / soul
    "r&b", "soul", "funk", "neo soul",
    # Jazz / blues
    "jazz", "blues",
    # Classical / cinematic
    "classical", "orchestral", "cinematic",
    # World / other
    "reggae", "latin", "afrobeat", "folk", "country",
]
MELODY = [
    "repetitive", "melodic", "complex", "simple", "driving",
    "catchy", "atonal", "chromatic", "pentatonic", "modal",
    "anthemic", "hypnotic", "sparse", "layered", "evolving",
]

ALL_LABELS = INSTRUMENTS + MOOD + ENERGY + GENRE + MELODY
assert len(ALL_LABELS) == 157, f"Expected 157 labels, got {len(ALL_LABELS)}"


def get_cache_dir() -> Path:
    home = Path.home()
    cache = home / ".cache" / "audio-track-comparator" / "clap"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def export_audio_encoder(model, cache_dir: Path) -> Path:
    """Export CLAP audio encoder using torch.jit.trace + legacy ONNX export."""
    import torch

    audio_onnx_path = cache_dir / "clap-htsat-unfused-audio.onnx"
    if audio_onnx_path.exists():
        print(f"Audio encoder already exists: {audio_onnx_path}")
        return audio_onnx_path

    print("Exporting audio encoder ONNX ...")

    class AudioEncoderWrapper(torch.nn.Module):
        def __init__(self, clap_model):
            super().__init__()
            self.audio_model = clap_model.audio_model
            self.audio_projection = clap_model.audio_projection

        def forward(self, mel):
            # mel: [batch, 1, 1001, 64]  (batch, channels, time_frames, n_mels)
            is_longer = torch.zeros(mel.shape[0], 1, dtype=torch.bool)
            outputs = self.audio_model(input_features=mel, is_longer=is_longer)
            pooled = outputs.pooler_output
            projected = self.audio_projection(pooled)
            return projected

    wrapper = AudioEncoderWrapper(model)
    wrapper.eval()

    # Correct input shape: [batch, channels, time_frames, n_mels] = [1, 1, 1001, 64]
    dummy = torch.zeros(1, 1, 1001, 64)

    # Verify forward pass
    with torch.no_grad():
        out = wrapper(dummy)
    assert out.shape == (1, 512), f"Unexpected output shape: {out.shape}"
    print(f"  Forward pass OK: output shape {out.shape}")

    # Trace then export using legacy path (avoids torch.export issues)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy,))

    try:
        from torch.onnx.utils import _export
        with torch.no_grad():
            _export(
                traced,
                (dummy,),
                str(audio_onnx_path),
                input_names=["mel_spectrogram"],
                output_names=["audio_embedding"],
                opset_version=14,
                do_constant_folding=True,
            )
        size_mb = audio_onnx_path.stat().st_size / 1e6
        print(f"  Audio encoder exported: {audio_onnx_path} ({size_mb:.1f} MB)")
        return audio_onnx_path
    except Exception as e:
        print(f"  Export failed: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        sys.exit(1)


def export_text_encoder(model, processor, cache_dir: Path) -> Path:
    """Export CLAP text encoder (optional — only used for future direct use)."""
    import torch

    text_onnx_path = cache_dir / "clap-htsat-unfused-text.onnx"
    if text_onnx_path.exists():
        print(f"Text encoder already exists: {text_onnx_path}")
        return text_onnx_path

    print("Exporting text encoder ONNX ...")

    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, clap_model):
            super().__init__()
            self.text_model = clap_model.text_model
            self.text_projection = clap_model.text_projection

        def forward(self, input_ids, attention_mask):
            outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.pooler_output
            projected = self.text_projection(pooled)
            return projected

    text_wrapper = TextEncoderWrapper(model)
    text_wrapper.eval()
    dummy_inputs = processor(text=["test audio"], return_tensors="pt", padding=True)
    dummy_ids = dummy_inputs["input_ids"]
    dummy_mask = dummy_inputs["attention_mask"]

    with torch.no_grad():
        traced_text = torch.jit.trace(text_wrapper, (dummy_ids, dummy_mask))

    try:
        from torch.onnx.utils import _export
        with torch.no_grad():
            _export(
                traced_text,
                (dummy_ids, dummy_mask),
                str(text_onnx_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["text_embedding"],
                opset_version=14,
                do_constant_folding=True,
            )
        size_mb = text_onnx_path.stat().st_size / 1e6
        print(f"  Text encoder exported: {text_onnx_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  Text encoder export failed (non-fatal): {type(e).__name__}", file=sys.stderr)
        text_onnx_path.write_bytes(b"placeholder")

    return text_onnx_path


def compute_text_embeddings(model, processor, cache_dir: Path) -> Path:
    """Pre-compute and save L2-normalized text embeddings for all 98 labels."""
    import torch
    import numpy as np

    text_emb_path = cache_dir / "text_embeddings.bin"
    if text_emb_path.exists():
        print(f"Text embeddings already exist: {text_emb_path}")
        return text_emb_path

    print(f"Computing text embeddings for {len(ALL_LABELS)} labels ...")

    batch_size = 16
    all_embeds = []

    for i in range(0, len(ALL_LABELS), batch_size):
        batch = ALL_LABELS[i : i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeds = model.get_text_features(**inputs)
        all_embeds.append(embeds.detach().cpu().numpy())
        print(f"  Labels {i}..{min(i + batch_size, len(ALL_LABELS))}")

    embeddings = np.concatenate(all_embeds, axis=0)  # [98, 512]
    assert embeddings.shape == (len(ALL_LABELS), 512), f"Unexpected shape: {embeddings.shape}"

    # L2-normalize each row
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    embeddings = (embeddings / norms).astype(np.float32)

    with open(text_emb_path, "wb") as f:
        f.write(embeddings.tobytes())

    size_kb = text_emb_path.stat().st_size / 1024
    print(f"  Text embeddings saved: {text_emb_path} ({size_kb:.1f} KB)")
    return text_emb_path


def main():
    try:
        import torch
        import numpy as np
        from transformers import ClapModel, ClapProcessor
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}", file=sys.stderr)
        print("Install with: pip install torch transformers numpy", file=sys.stderr)
        sys.exit(1)

    cache_dir = get_cache_dir()
    print(f"Cache directory: {cache_dir}")
    print()

    model_name = "laion/clap-htsat-unfused"
    print(f"Loading model: {model_name} ...")
    model = ClapModel.from_pretrained(model_name)
    processor = ClapProcessor.from_pretrained(model_name)
    model.eval()
    print("Model loaded.")
    print()

    audio_onnx_path = export_audio_encoder(model, cache_dir)
    print()
    text_onnx_path = export_text_encoder(model, processor, cache_dir)
    print()
    text_emb_path = compute_text_embeddings(model, processor, cache_dir)
    print()

    print("=" * 50)
    print("Export complete.")
    print(f"  Audio encoder: {audio_onnx_path.stat().st_size / 1e6:.2f} MB")
    print(f"  Text encoder:  {text_onnx_path.stat().st_size / 1e6:.2f} MB")
    print(f"  Text embeddings: {text_emb_path.stat().st_size / 1024:.1f} KB")
    print()
    print("Run the comparator:")
    print('  audio-track-comparator --reference "path/to/reference.mp3" --suno "path/to/suno.mp3"')


if __name__ == "__main__":
    main()

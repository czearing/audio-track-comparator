#!/usr/bin/env python3
"""
Evaluation script: BS-RoFormer-style stem separation (htdemucs_6s via demucs)
followed by LP-MusicCaps captioning per active stem.

This is a throwaway evaluation script. It does NOT integrate into the Rust pipeline.

Install requirements before running:
    pip install demucs transformers torch torchaudio

Usage:
    python scripts/test_stem_captioning.py
"""

import io
import os
import sys
import json
import shutil
import tempfile
import warnings
import subprocess
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths to test files
# ---------------------------------------------------------------------------
TEST_FILES = [
    Path("C:/Users/CaLebWork/Downloads/Knife Party - Bonfire.mp3"),
    Path("C:/Users/CaLebWork/Downloads/song.mp3"),
]

# Current MusiCNN instrument output per file (from most recent comparison JSON).
# Keys match the file stems so we can look them up at print time.
CURRENT_INSTRUMENT_OUTPUT = {
    "Knife Party - Bonfire": "techno, fast, loud, electronic, dance",
    "song": "techno, fast, electronic, dance, beat",
}

ENERGY_THRESHOLD = 0.05  # 5 % relative RMS
STEM_NAMES = ["drums", "bass", "other", "vocals", "guitar", "piano"]

# ---------------------------------------------------------------------------
# LP-MusicCaps model classes (inlined from the HuggingFace demo space so we
# do not need to clone the repo or install gradio).
# Source: https://huggingface.co/spaces/seungheondoh/LP-Music-Caps-demo
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig


def _sinusoids(length: int, channels: int, max_timescale: int = 10000) -> torch.Tensor:
    log_ts_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_ts = torch.exp(-log_ts_increment * torch.arange(channels // 2))
    scaled = torch.arange(length)[:, np.newaxis] * inv_ts[np.newaxis, :]
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)


class _MelEncoder(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 160,
        n_mels: int = 128,
    ) -> None:
        super().__init__()
        self.spec_fn = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=None,
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels,
            sample_rate,
            f_min,
            f_max,
            n_fft // 2 + 1,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self.spec_fn(wav)
        power_spec = spec.real.abs().pow(2)
        mel_spec = self.mel_scale(power_spec)
        return self.amplitude_to_db(mel_spec)


class _AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        audio_dim: int,
        text_dim: int,
        num_of_stride_conv: int,
    ) -> None:
        super().__init__()
        self.mel_encoder = _MelEncoder(n_mels=n_mels)
        self.conv1 = nn.Conv1d(n_mels, audio_dim, kernel_size=3, padding=1)
        self.conv_stack = nn.ModuleList(
            [
                nn.Conv1d(audio_dim, audio_dim, kernel_size=3, stride=2, padding=1)
                for _ in range(num_of_stride_conv)
            ]
        )
        self.register_buffer("positional_embedding", _sinusoids(n_ctx, text_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mel_encoder(x)
        x = F.gelu(self.conv1(x))
        for conv in self.conv_stack:
            x = F.gelu(conv(x))
        x = x.permute(0, 2, 1)
        return (x + self.positional_embedding).to(x.dtype)


class _BartCaptionModel(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        num_of_conv: int = 6,
        sr: int = 16000,
        duration: int = 10,
        max_length: int = 128,
        bart_type: str = "facebook/bart-base",
        audio_dim: int = 768,
    ) -> None:
        super().__init__()
        bart_config = BartConfig.from_pretrained(bart_type)
        self.tokenizer = BartTokenizer.from_pretrained(bart_type)
        self.bart = BartForConditionalGeneration(bart_config)

        self.n_sample = sr * duration
        self.hop_length = int(0.01 * sr)
        self.n_frames = int(self.n_sample // self.hop_length)
        num_of_stride_conv = num_of_conv - 1
        n_ctx = int(self.n_frames // 2 ** num_of_stride_conv) + 1

        self.audio_encoder = _AudioEncoder(
            n_mels=n_mels,
            n_ctx=n_ctx,
            audio_dim=audio_dim,
            text_dim=self.bart.config.hidden_size,
            num_of_stride_conv=num_of_stride_conv,
        )
        self.max_length = max_length

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def generate(
        self,
        samples: torch.Tensor,
        num_beams: int = 5,
        max_length: int = 128,
        min_length: int = 2,
        repetition_penalty: float = 1.0,
    ):
        audio_embs = self.audio_encoder(samples)
        encoder_outputs = self.bart.model.encoder(
            input_ids=None,
            attention_mask=None,
            inputs_embeds=audio_embs,
            return_dict=True,
        )
        input_ids = torch.zeros(
            (encoder_outputs["last_hidden_state"].size(0), 1), dtype=torch.long
        ).to(self.device)
        input_ids[:, 0] = self.bart.config.decoder_start_token_id
        decoder_attention_mask = torch.ones_like(input_ids)
        outputs = self.bart.generate(
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Helper: load LP-MusicCaps checkpoint
# ---------------------------------------------------------------------------

def _load_caption_model() -> _BartCaptionModel:
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download("seungheondoh/lp-music-caps", "transfer.pth")
    model = _BartCaptionModel(max_length=128)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Helper: load audio as mono float32 numpy array at target_sr
# ---------------------------------------------------------------------------

def _load_mono_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return audio


# ---------------------------------------------------------------------------
# Helper: RMS energy
# ---------------------------------------------------------------------------

def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


# ---------------------------------------------------------------------------
# Helper: separate stems via demucs htdemucs_6s
# Returns dict[stem_name -> np.ndarray] at 44100 Hz (demucs native SR)
# ---------------------------------------------------------------------------

def _separate_stems(audio_path: Path, tmp_dir: Path) -> dict:
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    import librosa
    import torchaudio.transforms as ta_transforms

    print("  Loading htdemucs_6s model ...")
    model = get_model("htdemucs_6s")
    model.eval()

    print("  Loading audio for separation ...")
    # Load with librosa (no torchcodec dependency), resample to model.samplerate
    audio_np, orig_sr = librosa.load(str(audio_path), sr=None, mono=False)
    # librosa returns (channels, samples) if mono=False, or (samples,) if mono
    if audio_np.ndim == 1:
        audio_np = np.stack([audio_np, audio_np])  # mono -> stereo (2, T)
    elif audio_np.shape[0] == 1:
        audio_np = np.concatenate([audio_np, audio_np], axis=0)  # (2, T)
    elif audio_np.shape[0] > 2:
        audio_np = audio_np[:2]

    if orig_sr != model.samplerate:
        left = librosa.resample(audio_np[0], orig_sr=orig_sr, target_sr=model.samplerate)
        right = librosa.resample(audio_np[1], orig_sr=orig_sr, target_sr=model.samplerate)
        audio_np = np.stack([left, right])

    waveform = torch.from_numpy(audio_np.astype(np.float32))

    # (batch, channels, samples)
    mix = waveform.unsqueeze(0)

    print("  Running demucs separation ...")
    with torch.no_grad():
        sources = apply_model(model, mix, device="cpu", progress=True)
    # sources shape: (batch, stems, channels, samples)
    sources = sources.squeeze(0)  # (stems, channels, samples)

    stems = {}
    for i, name in enumerate(model.sources):
        # average channels -> mono numpy
        stems[name] = sources[i].mean(0).numpy()

    return stems, model.samplerate


# ---------------------------------------------------------------------------
# Helper: prepare audio tensor for LP-MusicCaps (chunks of 10 s at 16 kHz)
# ---------------------------------------------------------------------------

def _prepare_caption_tensor(
    audio_16k: np.ndarray,
    duration_per_chunk: int = 10,
    target_sr: int = 16000,
) -> torch.Tensor:
    n_samples = int(duration_per_chunk * target_sr)
    if audio_16k.shape[0] < n_samples:
        pad = np.zeros(n_samples, dtype=np.float32)
        pad[: audio_16k.shape[0]] = audio_16k
        audio_16k = pad
    ceil = audio_16k.shape[0] // n_samples
    chunks = np.stack(
        np.split(audio_16k[: ceil * n_samples], ceil)
    ).astype(np.float32)
    return torch.from_numpy(chunks)


# ---------------------------------------------------------------------------
# Main per-file processing
# ---------------------------------------------------------------------------

def process_file(audio_path: Path, caption_model: _BartCaptionModel) -> None:
    file_key = audio_path.stem  # e.g. "Knife Party - Bonfire" or "song"

    print("=" * 60)
    print(f"TEST: {audio_path.name}")
    print("=" * 60)

    tmp_dir = Path(tempfile.mkdtemp(prefix="stem_cap_"))
    try:
        # ------------------------------------------------------------------
        # 1. Stem separation
        # ------------------------------------------------------------------
        print("\nSeparating stems with htdemucs_6s ...")
        stems_native, native_sr = _separate_stems(audio_path, tmp_dir)

        # Resample mix and each stem to 16 kHz for energy + captioning
        print("  Resampling to 16 kHz for analysis ...")
        import librosa

        mix_16k = _load_mono_audio(audio_path, target_sr=16000)
        mix_rms = _rms(mix_16k)

        stems_16k = {}
        for name, audio_native in stems_native.items():
            # audio_native is at native_sr; resample to 16k
            stem_16k = librosa.resample(
                audio_native.astype(np.float32),
                orig_sr=native_sr,
                target_sr=16000,
            )
            stems_16k[name] = stem_16k

        # ------------------------------------------------------------------
        # 2. Energy filter
        # ------------------------------------------------------------------
        print("\nSTEM ENERGY ANALYSIS")
        active_stems = []
        for name in STEM_NAMES:
            if name not in stems_16k:
                continue
            stem_rms = _rms(stems_16k[name])
            rel_pct = (stem_rms / mix_rms * 100.0) if mix_rms > 0 else 0.0
            if rel_pct >= ENERGY_THRESHOLD * 100:
                status = "ACTIVE"
                active_stems.append(name)
            else:
                status = f"SKIPPED (below threshold)"
            print(f"  {name:<8} {rel_pct:5.1f}%  -> {status}")

        # ------------------------------------------------------------------
        # 3. Current instrument output (MusiCNN)
        # ------------------------------------------------------------------
        print("\nCURRENT INSTRUMENT OUTPUT (MusiCNN):")
        current = CURRENT_INSTRUMENT_OUTPUT.get(file_key, "not available")
        print(f"  {current}")

        # ------------------------------------------------------------------
        # 4. LP-MusicCaps captions per active stem
        # ------------------------------------------------------------------
        print("\nLP-MUSICCAPS CAPTIONS PER STEM:")
        if not active_stems:
            print("  (no active stems above energy threshold)")
        else:
            for name in active_stems:
                audio = stems_16k[name]
                tensor = _prepare_caption_tensor(audio)
                with torch.no_grad():
                    captions = caption_model.generate(tensor, num_beams=5)

                # Combine all 10-second chunk captions into one string per stem
                combined = " | ".join(captions)
                print(f"  [{name}]  -> \"{combined}\"")

        print()
        print("=" * 60)
        print("VERDICT: manually evaluate whether captions are accurate")
        print("=" * 60)
        print()

    finally:
        # ------------------------------------------------------------------
        # 5. Cleanup temp files
        # ------------------------------------------------------------------
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Validate test files exist
    for p in TEST_FILES:
        if not p.exists():
            print(f"ERROR: test file not found: {p}", file=sys.stderr)
            sys.exit(1)

    print("Loading LP-MusicCaps (transfer checkpoint) ...")
    caption_model = _load_caption_model()
    print("Model loaded.\n")

    for audio_path in TEST_FILES:
        process_file(audio_path, caption_model)


if __name__ == "__main__":
    main()

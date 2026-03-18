# audio-track-comparator

A Rust CLI tool that compares a reference MP3 and a Suno-generated MP3, computing BPM, musical key, and CLAP-based semantic tags (instruments, mood, energy, genre, melody character) for each track, and outputs a structured JSON diff.

Designed for the Suno prompt refinement loop: generate → compare → refine prompt → regenerate.

## Prerequisites

- **Rust** (stable) — https://rustup.rs/
- **Python 3.8+** with pip — for the one-time model export
- ~2.5 GB disk space for the CLAP model cache

## One-Time Setup: Export CLAP Model

Before first use, run the export script to download `laion/clap-htsat-unfused` from HuggingFace and export it to ONNX:

```bash
pip install torch transformers numpy
python scripts/export_clap_onnx.py
```

This exports three files to `~/.cache/audio-track-comparator/clap/`:
- `clap-htsat-unfused-audio.onnx` — audio encoder (~117 MB)
- `clap-htsat-unfused-text.onnx` — text encoder (~501 MB)
- `text_embeddings.bin` — pre-computed text embeddings for all 98 vocabulary labels (~196 KB)

The export script only needs to run once. Subsequent runs use the cached files.

## Build

```bash
cd audio-track-comparator
cargo build --release
```

On Windows with Git for Windows (bash), the MSVC toolchain linker must be specified to avoid a conflict with GNU coreutils' `link` command. The `.cargo/config.toml` handles this automatically if Visual Studio Build Tools 2022 are installed at the default path.

If you have a different MSVC version, update `.cargo/config.toml`:
```toml
[target.x86_64-pc-windows-msvc]
linker = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Tools\\MSVC\\<version>\\bin\\Hostx64\\x64\\link.exe"
```

## Usage

```bash
audio-track-comparator \
  --reference "path/to/reference.mp3" \
  --suno "path/to/suno.mp3"
```

Optional flags:
- `--output <dir>` — write JSON to this directory (default: `output/` relative to CWD)
- `--no-cache-download` — exit with error if CLAP model files are not already cached

### Example

```bash
audio-track-comparator \
  --reference "C:/Users/CaLebWork/Downloads/Knife Party - Bonfire.mp3" \
  --suno "C:/Users/CaLebWork/Downloads/song.mp3"
```

Output (stdout summary printed first, then JSON file path):

```
=== Audio Track Comparison ===

                 Reference          Suno
  BPM:           128.0              130.5
  BPM source:    detected           detected
  Key:           A minor            A minor
  Energy:        high energy        high energy
  Top genre:     dubstep            edm

  BPM delta:     2.5
  Key match:     true (distance: 0 semitones)
  Energy match:  true

Report written to: output/Knife Party - Bonfire_vs_song_20260318_142201.json
```

## Output JSON Schema

See `dev-team-output/audio-track-comparator/requirements.md` Appendix B for the full schema.

Top-level structure:
```json
{
  "meta": { "analyzed_at", "reference_file", "suno_file" },
  "reference": { "bpm_bpm", "bpm_source", "key", "tags", "melody" },
  "suno": { ... },
  "diff": { "bpm", "key", "tags", "melody" }
}
```

## Label Vocabulary

The tool uses a fixed 98-label vocabulary split across 5 categories:
- 30 instrument labels
- 17 mood labels
- 4 energy labels
- 32 genre labels
- 15 melody descriptor labels

See `src/vocab.rs` for the full list.

## Architecture

```
src/
  main.rs         — CLI argument parsing and top-level orchestration
  decode.rs       — MP3 decoding via symphonia
  resample.rs     — Resampling via rubato (to 22050 Hz for BPM/key, to 48000 Hz for CLAP)
  bpm.rs          — BPM detection via autocorrelation on onset strength envelope
  key.rs          — Key detection via chroma + Krumhansl-Schmuckler
  clap_model.rs   — CLAP mel spectrogram + ONNX inference + top-N tag selection
  model_cache.rs  — Cache path management with setup instructions
  pipeline.rs     — Per-file analysis orchestration
  diff.rs         — Structured diff computation
  output.rs       — JSON serialization and terminal summary
  vocab.rs        — Label vocabulary constants

scripts/
  export_clap_onnx.py  — One-time ONNX export and text embedding pre-computation
```

## Performance

On a modern CPU with the CLAP model cached, typical run time is 15–45 seconds for two 4-minute MP3 files.

The ONNX runtime (`ort`) uses all available CPU threads. Set `intra_op_num_threads` manually via the `ORT_NUM_THREADS` environment variable if you want to limit CPU usage.

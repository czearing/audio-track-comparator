/// Discogs-EffNet contrastive similarity.
///
/// Model: similarity.onnx  (discogs_multi_embeddings-effnet-bs64-1.onnx)
///   Input:  [64, 128, 96]  float32  (fixed batch of 64 mel patches)
///   Output: [64, 1280]     float32  (contrastive embeddings per patch)
///
/// Trained with a contrastive objective grouping tracks by artist and track
/// associations — purpose-built for music similarity, not audio event
/// classification. Preprocessing is identical to the Discogs-EffNet
/// classification backbone (backbone.rs):
///   sample_rate = 16 000 Hz
///   frame_size  = 512 samples
///   hop_size    = 256 samples
///   n_mels      = 96
///   patch_size  = 128 frames
use crate::{decode, resample};
use anyhow::Context;
use ndarray::Array3;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

const SAMPLE_RATE: u32 = 16_000;
const FRAME_SIZE: usize = 512;
const HOP_SIZE: usize = 256;
const N_MELS: usize = 96;
const PATCH_SIZE: usize = 128;
const N_EMBED: usize = 1280;
const BATCH_SIZE: usize = 64; // model has fixed batch size
const F_MIN: f64 = 0.0;
const F_MAX: f64 = 8_000.0;

fn build_mel_filterbank() -> Vec<Vec<f32>> {
    let n_freqs = FRAME_SIZE / 2 + 1;
    let sr = SAMPLE_RATE as f64;

    fn hz_to_mel(f: f64) -> f64 {
        2595.0 * (1.0 + f / 700.0).log10()
    }
    fn mel_to_hz(m: f64) -> f64 {
        700.0 * (10.0f64.powf(m / 2595.0) - 1.0)
    }

    let mel_min = hz_to_mel(F_MIN);
    let mel_max = hz_to_mel(F_MAX);
    let n_pts = N_MELS + 2;
    let mel_points: Vec<f64> = (0..n_pts)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_pts - 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * sr / FRAME_SIZE as f64)
        .collect();

    let mut fb: Vec<Vec<f32>> = Vec::with_capacity(N_MELS);
    for m in 0..N_MELS {
        let lower = hz_points[m];
        let center = hz_points[m + 1];
        let upper = hz_points[m + 2];
        let norm = if upper > lower { 2.0 / (upper - lower) } else { 1.0 };
        let mut row = vec![0.0f32; n_freqs];
        for k in 0..n_freqs {
            let f = fft_freqs[k];
            row[k] = if f <= lower || f >= upper {
                0.0
            } else if f <= center {
                (norm * (f - lower) / (center - lower)) as f32
            } else {
                (norm * (upper - f) / (upper - center)) as f32
            };
        }
        fb.push(row);
    }
    fb
}

#[inline]
fn log_compression(x: f32) -> f32 {
    (10000.0 * x + 1.0).log10()
}

fn compute_patches(mono_16k: &[f32]) -> Vec<Vec<f32>> {
    use rustfft::{num_complex::Complex, FftPlanner};

    let n_freqs = FRAME_SIZE / 2 + 1;
    let filterbank = build_mel_filterbank();
    let window: Vec<f32> = (0..FRAME_SIZE)
        .map(|i| {
            0.5 * (1.0
                - (2.0 * std::f32::consts::PI * i as f32 / (FRAME_SIZE - 1) as f32).cos())
        })
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FRAME_SIZE);

    let n_frames = if mono_16k.len() >= FRAME_SIZE {
        (mono_16k.len() - FRAME_SIZE) / HOP_SIZE + 1
    } else {
        0
    };
    if n_frames == 0 {
        return Vec::new();
    }

    let mut mel_frames: Vec<Vec<f32>> = Vec::with_capacity(n_frames);
    let mut buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); FRAME_SIZE];
    for frame_idx in 0..n_frames {
        let start = frame_idx * HOP_SIZE;
        for i in 0..FRAME_SIZE {
            let s = *mono_16k.get(start + i).unwrap_or(&0.0);
            buf[i] = Complex::new(s * window[i], 0.0);
        }
        fft.process(&mut buf);
        let power: Vec<f32> = (0..n_freqs).map(|k| buf[k].norm_sqr()).collect();
        let mel: Vec<f32> = filterbank
            .iter()
            .map(|row| {
                let v: f32 = row.iter().zip(power.iter()).map(|(&w, &p)| w * p).sum();
                log_compression(v)
            })
            .collect();
        mel_frames.push(mel);
    }

    let n_patches = n_frames / PATCH_SIZE;
    if n_patches == 0 {
        return Vec::new();
    }

    let mut patches: Vec<Vec<f32>> = Vec::with_capacity(n_patches);
    for p in 0..n_patches {
        let mut patch = Vec::with_capacity(PATCH_SIZE * N_MELS);
        for f in 0..PATCH_SIZE {
            patch.extend_from_slice(&mel_frames[p * PATCH_SIZE + f]);
        }
        patches.push(patch);
    }
    patches
}

/// Sample up to BATCH_SIZE patches evenly from the track, pad to exactly
/// BATCH_SIZE with zeros if fewer patches exist. Returns the flat batch
/// and the number of real (non-padded) patches.
fn prepare_batch(patches: Vec<Vec<f32>>) -> (Vec<f32>, usize) {
    let _n_real = patches.len().min(BATCH_SIZE);
    let patch_len = PATCH_SIZE * N_MELS;

    // Evenly sample n_real indices from patches
    let selected: Vec<Vec<f32>> = if patches.len() <= BATCH_SIZE {
        patches
    } else {
        (0..BATCH_SIZE)
            .map(|i| {
                let idx = i * (patches.len() - 1) / (BATCH_SIZE - 1);
                patches[idx].clone()
            })
            .collect()
    };

    let n_selected = selected.len();
    let mut flat = vec![0.0f32; BATCH_SIZE * patch_len];
    for (i, patch) in selected.into_iter().enumerate() {
        flat[i * patch_len..(i + 1) * patch_len].copy_from_slice(&patch);
    }

    (flat, n_selected)
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Run the contrastive model on a batch, return mean-pooled [N_EMBED] embedding
/// using only the first `n_real` rows of the output.
fn embed(mono_22050: &[f32], onnx_path: &Path) -> anyhow::Result<Vec<f32>> {
    let mono_16k = resample::resample(mono_22050, 22050, SAMPLE_RATE)
        .context("Failed to resample to 16kHz for similarity model")?;

    let patches = compute_patches(&mono_16k);
    if patches.is_empty() {
        anyhow::bail!("Audio too short to compute similarity patches");
    }

    let (flat, n_real) = prepare_batch(patches);

    let arr = Array3::<f32>::from_shape_vec((BATCH_SIZE, PATCH_SIZE, N_MELS), flat)
        .context("Failed to shape similarity input tensor")?;

    let mut session = Session::builder()
        .map_err(|e| anyhow::anyhow!("ORT session builder error: {}", e))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| anyhow::anyhow!("ORT optimization level error: {}", e))?
        .with_intra_threads(num_cpus())
        .map_err(|e| anyhow::anyhow!("ORT thread count error: {}", e))?
        .commit_from_file(onnx_path)
        .map_err(|e| anyhow::anyhow!("Failed to load similarity ONNX model: {}", e))?;

    let tensor = Tensor::<f32>::from_array(arr)
        .map_err(|e| anyhow::anyhow!("Failed to create similarity input tensor: {}", e))?;

    let outputs = session
        .run(ort::inputs![tensor])
        .map_err(|e| anyhow::anyhow!("Similarity ONNX inference failed: {}", e))?;

    let (_shape, raw) = outputs[0_usize]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract similarity output tensor: {}", e))?;

    if raw.len() != BATCH_SIZE * N_EMBED {
        anyhow::bail!(
            "Similarity model output size mismatch: expected {}, got {}",
            BATCH_SIZE * N_EMBED,
            raw.len()
        );
    }

    // Mean-pool only over real (non-padded) patches
    let mut mean_embed = vec![0.0f32; N_EMBED];
    for p in 0..n_real {
        for d in 0..N_EMBED {
            mean_embed[d] += raw[p * N_EMBED + d];
        }
    }
    let n_f = n_real as f32;
    for v in mean_embed.iter_mut() {
        *v /= n_f;
    }

    Ok(mean_embed)
}

/// Compute Discogs-EffNet contrastive similarity between two tracks.
/// Returns a score in [0.0, 100.0] rounded to one decimal place.
pub fn compute(ref_path: &Path, suno_path: &Path, onnx_path: &Path) -> anyhow::Result<f32> {
    let ref_buf = decode::decode_mp3(ref_path)
        .with_context(|| format!("Failed to decode reference: {}", ref_path.display()))?;
    let ref_mono = resample::to_mono(&ref_buf);

    let suno_buf = decode::decode_mp3(suno_path)
        .with_context(|| format!("Failed to decode suno: {}", suno_path.display()))?;
    let suno_mono = resample::to_mono(&suno_buf);

    // Resample both to 22050 Hz (embed() resamples internally to 16kHz)
    let ref_22k = resample::resample(&ref_mono, ref_buf.sample_rate, 22050)
        .context("Failed to resample reference to 22050 Hz")?;
    let suno_22k = resample::resample(&suno_mono, suno_buf.sample_rate, 22050)
        .context("Failed to resample suno to 22050 Hz")?;

    let ref_emb = embed(&ref_22k, onnx_path).context("Failed to embed reference track")?;
    let suno_emb = embed(&suno_22k, onnx_path).context("Failed to embed suno track")?;

    let ref_norm: f32 = ref_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    let suno_norm: f32 = suno_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if ref_norm < 1e-8 {
        anyhow::bail!("Reference audio is silent — embedding norm is zero");
    }
    if suno_norm < 1e-8 {
        anyhow::bail!("Suno audio is silent — embedding norm is zero");
    }

    let dot: f32 = ref_emb.iter().zip(suno_emb.iter()).map(|(a, b)| a * b).sum();
    let similarity = dot / (ref_norm * suno_norm);

    let score = (similarity * 100.0).clamp(0.0, 100.0);
    Ok((score * 10.0).round() / 10.0)
}

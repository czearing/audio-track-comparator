/// VGGish-based acoustic similarity.
///
/// Model: vggish.onnx
///   Input:  [N, 1, 96, 64]  float32  (N patches, 1 channel, 96 time frames, 64 mel bins)
///   Output: [N, 128]     float32  (one 128-dim embedding per patch)
///
/// Preprocessing (matches VGGish original):
///   sample_rate = 16 000 Hz
///   window      = 25 ms Hann  (400 samples)
///   hop         = 10 ms       (160 samples)
///   n_mels      = 64
///   f_min       = 125 Hz
///   f_max       = 7 500 Hz
///   patch_size  = 96 frames   (non-overlapping — patch hop equals patch size)
///   log         = log10(x + 0.01)
use crate::{decode, resample};
use anyhow::Context;
use ndarray::Array4;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

const VGGISH_SAMPLE_RATE: u32 = 16_000;
const FRAME_SIZE: usize = 400; // 25 ms at 16 kHz
const HOP_SIZE: usize = 160; // 10 ms at 16 kHz
const N_MELS: usize = 64;
const PATCH_SIZE: usize = 96; // time frames per patch
const F_MIN: f64 = 125.0;
const F_MAX: f64 = 7_500.0;

/// Build a linear-scale mel filterbank [N_MELS, FRAME_SIZE/2+1].
/// Uses HTK-style Hz→mel conversion: mel = 2595 * log10(1 + f/700).
fn build_mel_filterbank() -> Vec<Vec<f32>> {
    let n_freqs = FRAME_SIZE / 2 + 1;
    let sr = VGGISH_SAMPLE_RATE as f64;

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
        // Slaney area normalization: 2 / (upper - lower) in Hz
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

/// VGGish log compression: log10(x + 0.01).
#[inline]
fn log_compression(x: f32) -> f32 {
    (x + 0.01).log10()
}

/// Compute VGGish log-mel patches from 16 kHz mono audio.
/// Returns a Vec of patches, each of shape [PATCH_SIZE × N_MELS] (row-major).
/// Patches are non-overlapping: patch hop equals PATCH_SIZE.
fn compute_patches(mono_16k: &[f32]) -> Vec<Vec<f32>> {
    use rustfft::{num_complex::Complex, FftPlanner};

    let n_freqs = FRAME_SIZE / 2 + 1;
    let filterbank = build_mel_filterbank();

    // Hann window
    let window: Vec<f32> = (0..FRAME_SIZE)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (FRAME_SIZE - 1) as f32).cos())
        })
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FRAME_SIZE);

    // Number of STFT frames
    let n_frames = if mono_16k.len() >= FRAME_SIZE {
        (mono_16k.len() - FRAME_SIZE) / HOP_SIZE + 1
    } else {
        0
    };

    if n_frames == 0 {
        return Vec::new();
    }

    // Compute mel frames: mel_frames[frame][mel_bin]
    let mut mel_frames: Vec<Vec<f32>> = Vec::with_capacity(n_frames);
    let mut buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); FRAME_SIZE];

    for frame_idx in 0..n_frames {
        let start = frame_idx * HOP_SIZE;
        for i in 0..FRAME_SIZE {
            let s = *mono_16k.get(start + i).unwrap_or(&0.0);
            buf[i] = Complex::new(s * window[i], 0.0);
        }
        fft.process(&mut buf);

        // Power spectrum [n_freqs]
        let power: Vec<f32> = (0..n_freqs).map(|k| buf[k].norm_sqr()).collect();

        // Project to mel and apply log compression
        let mel: Vec<f32> = filterbank
            .iter()
            .map(|row| {
                let v: f32 = row.iter().zip(power.iter()).map(|(&w, &p)| w * p).sum();
                log_compression(v)
            })
            .collect();

        mel_frames.push(mel);
    }

    // Split into non-overlapping patches of PATCH_SIZE frames
    if n_frames < PATCH_SIZE {
        return Vec::new();
    }

    let mut patches: Vec<Vec<f32>> = Vec::new();
    let mut start = 0;
    while start + PATCH_SIZE <= n_frames {
        // patch layout: [PATCH_SIZE, N_MELS] row-major
        let mut patch = Vec::with_capacity(PATCH_SIZE * N_MELS);
        for f in 0..PATCH_SIZE {
            patch.extend_from_slice(&mel_frames[start + f]);
        }
        patches.push(patch);
        start += PATCH_SIZE; // non-overlapping
    }
    patches
}

/// Compute the mean-pooled 128-dim VGGish embedding for a 16 kHz mono signal.
fn embed(mono_16k: &[f32], onnx_path: &Path) -> anyhow::Result<Vec<f32>> {
    let patches = compute_patches(mono_16k);

    if patches.is_empty() {
        anyhow::bail!("Audio too short for VGGish patches");
    }

    let n_patches = patches.len();

    // Build batch tensor [n_patches, 1, PATCH_SIZE, N_MELS] — VGGish expects a channel dim
    let flat: Vec<f32> = patches.into_iter().flatten().collect();
    let arr = Array4::<f32>::from_shape_vec((n_patches, 1, PATCH_SIZE, N_MELS), flat)
        .context("Failed to shape VGGish input tensor")?;

    // Load ONNX session
    let mut session = Session::builder()
        .map_err(|e| anyhow::anyhow!("Failed to create ORT session builder: {}", e))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?
        .with_intra_threads(num_cpus())
        .map_err(|e| anyhow::anyhow!("Failed to set thread count: {}", e))?
        .commit_from_file(onnx_path)
        .map_err(|e| anyhow::anyhow!("Failed to load VGGish ONNX model: {}", e))?;

    // Run inference
    let input_tensor = Tensor::<f32>::from_array(arr)
        .map_err(|e| anyhow::anyhow!("Failed to create VGGish input tensor: {}", e))?;

    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| anyhow::anyhow!("VGGish ONNX inference failed: {}", e))?;

    // Output 0 is [n_patches, 128] embeddings
    let (_shape, raw) = outputs[0_usize]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract VGGish output tensor: {}", e))?;

    const EMBED_DIM: usize = 128;
    let total_elements = raw.len();
    if total_elements != n_patches * EMBED_DIM {
        anyhow::bail!(
            "VGGish model output size mismatch: expected {}, got {}",
            n_patches * EMBED_DIM,
            total_elements
        );
    }

    // Mean-pool across patches
    let mut mean_embedding = vec![0.0f32; EMBED_DIM];
    for p in 0..n_patches {
        for d in 0..EMBED_DIM {
            mean_embedding[d] += raw[p * EMBED_DIM + d];
        }
    }
    let n_f = n_patches as f32;
    for v in mean_embedding.iter_mut() {
        *v /= n_f;
    }

    Ok(mean_embedding)
}

/// Compute cosine similarity between two VGGish embeddings, scaled to a percentage.
///
/// Returns a value in [0.0, 100.0] rounded to one decimal place.
pub fn compute(ref_path: &Path, suno_path: &Path, onnx_path: &Path) -> anyhow::Result<f32> {
    // Decode and resample reference
    let ref_buf = decode::decode_mp3(ref_path)
        .with_context(|| format!("Failed to decode reference: {}", ref_path.display()))?;
    let ref_mono = resample::to_mono(&ref_buf);
    let ref_16k = resample::resample(&ref_mono, ref_buf.sample_rate, VGGISH_SAMPLE_RATE)
        .context("Failed to resample reference to 16 kHz")?;

    // Decode and resample suno
    let suno_buf = decode::decode_mp3(suno_path)
        .with_context(|| format!("Failed to decode suno: {}", suno_path.display()))?;
    let suno_mono = resample::to_mono(&suno_buf);
    let suno_16k = resample::resample(&suno_mono, suno_buf.sample_rate, VGGISH_SAMPLE_RATE)
        .context("Failed to resample suno to 16 kHz")?;

    // Compute embeddings
    let ref_emb = embed(&ref_16k, onnx_path).context("Failed to embed reference track")?;
    let suno_emb = embed(&suno_16k, onnx_path).context("Failed to embed suno track")?;

    // Guard against silent / near-silent audio
    let ref_norm: f32 = ref_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    let suno_norm: f32 = suno_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if ref_norm < 1e-8 {
        anyhow::bail!("Reference audio is silent or near-silent — VGGish embedding is zero");
    }
    if suno_norm < 1e-8 {
        anyhow::bail!("Suno audio is silent or near-silent — VGGish embedding is zero");
    }

    // Cosine similarity
    let dot: f32 = ref_emb
        .iter()
        .zip(suno_emb.iter())
        .map(|(a, b)| a * b)
        .sum();
    let similarity = dot / (ref_norm * suno_norm);

    // Scale to percentage, clamp, and round to 1 decimal place
    let score = (similarity * 100.0).clamp(0.0, 100.0);
    let score = (score * 10.0).round() / 10.0;
    Ok(score)
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

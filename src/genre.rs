/// Discogs-EffNet genre classifier.
///
/// Model: discogs-effnet-bsdynamic-1.onnx
///   Input:  [n_patches, 128, 96]  float32   (n_patches × time_frames × mel_bins)
///   Output: [n_patches, 400]      float32   (sigmoid genre probabilities)
///
/// Preprocessing (matches Essentia TensorflowInputMusiCNN defaults):
///   sample_rate = 16 000 Hz
///   frame_size  = 512 samples  (FFT window)
///   hop_size    = 256 samples
///   n_mels      = 96
///   patch_size  = 128 frames   (one model input)
use crate::resample;
use anyhow::Context;
use ndarray::Array3;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

const GENRE_SAMPLE_RATE: u32 = 16_000;
const FRAME_SIZE: usize = 512;
const HOP_SIZE: usize = 256;
const N_MELS: usize = 96;
const PATCH_SIZE: usize = 128; // time frames per patch

// Mel filterbank frequency range (Essentia defaults for MusiCNN)
const F_MIN: f64 = 0.0;
const F_MAX: f64 = 8_000.0; // Nyquist at 16 kHz

/// Build a linear-scale mel filterbank [N_MELS, FRAME_SIZE/2+1].
/// Uses HTK-style Hz→mel conversion: mel = 2595 * log10(1 + f/700).
fn build_mel_filterbank() -> Vec<Vec<f32>> {
    let n_freqs = FRAME_SIZE / 2 + 1;
    let sr = GENRE_SAMPLE_RATE as f64;

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

/// LogC compression matching the MusiCNN/EffNet training preprocessing:
/// `log10(10000 * x + 1)` where x is the linear mel power value.
#[inline]
fn log_compression(x: f32) -> f32 {
    (10000.0 * x + 1.0).log10()
}

/// Compute log mel spectrogram from 16kHz mono audio.
/// Returns a Vec of patches, each of shape [PATCH_SIZE × N_MELS] (row-major).
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

    // Compute power spectrum and project to mel — one frame at a time to save memory
    // mel_frames[frame][mel_bin]
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

        // Project to mel and apply logC compression: log10(10000 * x + 1)
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
    let n_patches = n_frames / PATCH_SIZE;
    if n_patches == 0 {
        return Vec::new();
    }

    let mut patches: Vec<Vec<f32>> = Vec::with_capacity(n_patches);
    for p in 0..n_patches {
        // patch layout: [PATCH_SIZE, N_MELS] row-major
        let mut patch = Vec::with_capacity(PATCH_SIZE * N_MELS);
        for f in 0..PATCH_SIZE {
            patch.extend_from_slice(&mel_frames[p * PATCH_SIZE + f]);
        }
        patches.push(patch);
    }
    patches
}

fn top_n_indices(scores: &[f32], n: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(n).map(|(i, _)| *i).collect()
}

/// Detect top-5 genre tags from mono 22050Hz audio using the Discogs-EffNet model.
///
/// `cache_dir` must contain `discogs_genre.onnx` and `genre_labels.json`.
pub fn detect(mono_22050: &[f32], cache_dir: &Path) -> anyhow::Result<Vec<String>> {
    let onnx_path = cache_dir.join("discogs_genre.onnx");
    let labels_path = cache_dir.join("genre_labels.json");

    // Load labels
    let labels_json =
        std::fs::read_to_string(&labels_path).context("Failed to read genre_labels.json")?;
    let labels: Vec<String> =
        serde_json::from_str(&labels_json).context("Failed to parse genre_labels.json")?;
    if labels.len() != 400 {
        anyhow::bail!(
            "genre_labels.json: expected 400 labels, got {}",
            labels.len()
        );
    }

    // Resample 22050 → 16000 Hz
    let mono_16k = resample::resample(mono_22050, 22050, GENRE_SAMPLE_RATE)
        .context("Failed to resample to 16kHz for genre model")?;

    // Compute mel patches
    let patches = compute_patches(&mono_16k);
    if patches.is_empty() {
        // Not enough audio — return empty
        return Ok(Vec::new());
    }

    let n_patches = patches.len();

    // Build batch tensor [n_patches, PATCH_SIZE, N_MELS]
    let flat: Vec<f32> = patches.into_iter().flatten().collect();
    let arr = Array3::<f32>::from_shape_vec((n_patches, PATCH_SIZE, N_MELS), flat)
        .context("Failed to shape genre input tensor")?;

    // Load ONNX session
    let mut session = Session::builder()
        .map_err(|e| anyhow::anyhow!("Failed to create ORT session builder: {}", e))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?
        .with_intra_threads(num_cpus())
        .map_err(|e| anyhow::anyhow!("Failed to set thread count: {}", e))?
        .commit_from_file(&onnx_path)
        .map_err(|e| anyhow::anyhow!("Failed to load genre ONNX model: {}", e))?;

    // Run inference
    let input_tensor = Tensor::<f32>::from_array(arr)
        .map_err(|e| anyhow::anyhow!("Failed to create genre input tensor: {}", e))?;

    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| anyhow::anyhow!("Genre ONNX inference failed: {}", e))?;

    // Output 0 is [n_patches, 400] genre probabilities
    let (_shape, raw) = outputs[0_usize]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract genre output tensor: {}", e))?;

    let total_elements = raw.len();
    if total_elements != n_patches * 400 {
        anyhow::bail!(
            "Genre model output size mismatch: expected {}, got {}",
            n_patches * 400,
            total_elements
        );
    }

    // Average predictions across patches
    let mut avg_scores = vec![0.0f32; 400];
    for p in 0..n_patches {
        for g in 0..400 {
            avg_scores[g] += raw[p * 400 + g];
        }
    }
    let n_f = n_patches as f32;
    for s in avg_scores.iter_mut() {
        *s /= n_f;
    }

    // Return top-5 genre labels
    let top5 = top_n_indices(&avg_scores, 5);
    let result: Vec<String> = top5.into_iter().map(|i| labels[i].clone()).collect();
    Ok(result)
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Shared Discogs-EffNet backbone — runs once per track and produces both
/// genre probabilities and embeddings for all downstream head models.
///
/// Model: discogs_genre.onnx
///   Input:  [n_patches, 128, 96]  float32  (mel patches)
///   Output 0: [n_patches, 400]   float32  (sigmoid genre probabilities)
///   Output 1: [n_patches, 1280]  float32  (EffNet embeddings)
///
/// Preprocessing (Essentia TensorflowInputMusiCNN defaults):
///   sample_rate = 16 000 Hz
///   frame_size  = 512 samples
///   hop_size    = 256 samples
///   n_mels      = 96
///   patch_size  = 128 frames
use crate::resample;
use anyhow::Context;
use ndarray::Array3;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

const BACKBONE_SAMPLE_RATE: u32 = 16_000;
const FRAME_SIZE: usize = 512;
const HOP_SIZE: usize = 256;
const N_MELS: usize = 96;
const PATCH_SIZE: usize = 128;
const N_GENRE: usize = 400;
const N_EMBED: usize = 1280;

const F_MIN: f64 = 0.0;
const F_MAX: f64 = 8_000.0;

/// Output produced by the backbone for a single track.
pub struct BackboneOutput {
    /// Flattened [n_patches, 400] genre probabilities (sigmoid).
    pub genre_probs: Vec<f32>,
    /// Flattened [n_patches, 1280] EffNet embeddings.
    pub embeddings: Vec<f32>,
    pub n_patches: usize,
}

fn build_mel_filterbank() -> Vec<Vec<f32>> {
    let n_freqs = FRAME_SIZE / 2 + 1;
    let sr = BACKBONE_SAMPLE_RATE as f64;

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

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Run the Discogs-EffNet backbone once and return both genre probabilities
/// and embeddings for all downstream consumers.
///
/// `mono_22050` — mono audio at 22050 Hz.
/// `backbone_path` — path to `discogs_genre.onnx`.
pub fn run(mono_22050: &[f32], backbone_path: &Path) -> anyhow::Result<BackboneOutput> {
    // Resample 22050 → 16000 Hz
    let mono_16k = resample::resample(mono_22050, 22050, BACKBONE_SAMPLE_RATE)
        .context("Failed to resample to 16kHz for backbone")?;

    // Compute mel patches
    let patches = compute_patches(&mono_16k);
    if patches.is_empty() {
        return Ok(BackboneOutput {
            genre_probs: Vec::new(),
            embeddings: Vec::new(),
            n_patches: 0,
        });
    }

    let n_patches = patches.len();
    let flat: Vec<f32> = patches.into_iter().flatten().collect();
    let arr = Array3::<f32>::from_shape_vec((n_patches, PATCH_SIZE, N_MELS), flat)
        .context("Failed to shape backbone input tensor")?;

    let mut session = Session::builder()
        .map_err(|e| anyhow::anyhow!("Failed to create ORT session builder: {}", e))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?
        .with_intra_threads(num_cpus())
        .map_err(|e| anyhow::anyhow!("Failed to set thread count: {}", e))?
        .commit_from_file(backbone_path)
        .map_err(|e| anyhow::anyhow!("Failed to load backbone ONNX model: {}", e))?;

    let input_tensor = Tensor::<f32>::from_array(arr)
        .map_err(|e| anyhow::anyhow!("Failed to create backbone input tensor: {}", e))?;

    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| anyhow::anyhow!("Backbone ONNX inference failed: {}", e))?;

    // Output 0: [n_patches, 400] genre probabilities
    let (_shape, genre_raw) = outputs[0_usize]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract backbone genre output: {}", e))?;

    if genre_raw.len() != n_patches * N_GENRE {
        anyhow::bail!(
            "Backbone genre output size mismatch: expected {}, got {}",
            n_patches * N_GENRE,
            genre_raw.len()
        );
    }

    // Output 1: [n_patches, 1280] embeddings
    let (_shape2, embed_raw) = outputs[1_usize]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract backbone embeddings: {}", e))?;

    if embed_raw.len() != n_patches * N_EMBED {
        anyhow::bail!(
            "Backbone embedding size mismatch: expected {}, got {}",
            n_patches * N_EMBED,
            embed_raw.len()
        );
    }

    Ok(BackboneOutput {
        genre_probs: genre_raw.to_vec(),
        embeddings: embed_raw.to_vec(),
        n_patches,
    })
}

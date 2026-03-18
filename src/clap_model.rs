use crate::model_cache::ModelPaths;
use crate::vocab;
use anyhow::Context;
use ndarray::{Array1, Array2, Array4};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

pub struct Tags {
    /// Instrument tags sourced from the MTT MusiCNN model at runtime (owned Strings).
    pub instruments: Vec<String>,
    pub mood: Vec<&'static str>,
    pub energy: &'static str,
    /// Genre labels sourced from the Discogs-EffNet model at runtime (owned Strings).
    pub genre: Vec<String>,
}

pub struct Melody {
    pub descriptors: Vec<&'static str>,
}

// Mel spectrogram parameters matching ClapFeatureExtractor
const SAMPLE_RATE: usize = 48000;
const N_FFT: usize = 1024;
const HOP_LENGTH: usize = 480;
const N_MELS: usize = 64;
const F_MIN: f64 = 50.0;
const F_MAX: f64 = 14000.0;
const TARGET_FRAMES: usize = 1001;
const EMBED_DIM: usize = 512;

// Slaney mel scale parameters
const F_SP: f64 = 200.0 / 3.0; // ≈ 66.667
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP; // = 15.0

fn logstep() -> f64 {
    6.4f64.ln() / 27.0
}

fn hz_to_mel_slaney(freq: f64) -> f64 {
    if freq < MIN_LOG_HZ {
        freq / F_SP
    } else {
        MIN_LOG_MEL + (freq / MIN_LOG_HZ).ln() / logstep()
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * logstep()).exp()
    }
}

/// Build Slaney-norm mel filterbank matrix: shape [n_mels, n_fft/2+1]
fn build_mel_filterbank() -> Array2<f32> {
    let n_freqs = N_FFT / 2 + 1;
    let mel_min = hz_to_mel_slaney(F_MIN);
    let mel_max = hz_to_mel_slaney(F_MAX);

    let n_pts = N_MELS + 2;
    let mel_points: Vec<f64> = (0..n_pts)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_pts - 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz_slaney(m)).collect();

    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * SAMPLE_RATE as f64 / N_FFT as f64)
        .collect();

    let mut fb = Array2::<f32>::zeros((N_MELS, n_freqs));

    for m in 0..N_MELS {
        let lower = hz_points[m];
        let center = hz_points[m + 1];
        let upper = hz_points[m + 2];

        let norm = 2.0 / (upper - lower);

        for k in 0..n_freqs {
            let f = fft_freqs[k];
            let val = if f < lower || f > upper {
                0.0
            } else if f <= center {
                norm * (f - lower) / (center - lower)
            } else {
                norm * (upper - f) / (upper - center)
            };
            fb[[m, k]] = val as f32;
        }
    }

    fb
}

/// Compute mel spectrogram from 48kHz mono PCM. Returns [n_mels, n_frames].
fn compute_mel_spectrogram(mono_48000: &[f32]) -> Array2<f32> {
    use rustfft::{num_complex::Complex, FftPlanner};

    let filterbank = build_mel_filterbank();

    let window: Vec<f32> = (0..N_FFT)
        .map(|i| {
            0.5 * (1.0
                - (2.0 * std::f32::consts::PI * i as f32 / (N_FFT - 1) as f32).cos())
        })
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);

    let n_frames = if mono_48000.len() >= N_FFT {
        (mono_48000.len() - N_FFT) / HOP_LENGTH + 1
    } else {
        0
    };

    let n_freqs = N_FFT / 2 + 1;
    let mut power_spec = Array2::<f32>::zeros((n_freqs, n_frames));
    let mut buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); N_FFT];

    for frame_idx in 0..n_frames {
        let start = frame_idx * HOP_LENGTH;
        for i in 0..N_FFT {
            let s = if start + i < mono_48000.len() {
                mono_48000[start + i]
            } else {
                0.0
            };
            buf[i] = Complex::new(s * window[i], 0.0);
        }
        fft.process(&mut buf);
        for k in 0..n_freqs {
            power_spec[[k, frame_idx]] = buf[k].norm_sqr();
        }
    }

    // mel_spec = filterbank @ power_spec → [n_mels, n_frames]
    let mel_spec = filterbank.dot(&power_spec);

    // 10 * log10, floor at 1e-10
    mel_spec.mapv(|x| 10.0 * x.max(1e-10).log10())
}

/// Pad or truncate to TARGET_FRAMES and normalize.
/// Input: [n_mels, n_frames]. Output: [n_mels, TARGET_FRAMES]
fn normalize_frames(spec: Array2<f32>) -> Array2<f32> {
    let (n_mels, n_frames) = spec.dim();
    let mut out = Array2::<f32>::zeros((n_mels, TARGET_FRAMES));

    if n_frames >= TARGET_FRAMES {
        out.assign(&spec.slice(ndarray::s![.., ..TARGET_FRAMES]));
    } else {
        out.slice_mut(ndarray::s![.., ..n_frames]).assign(&spec);
    }

    let flat: Vec<f32> = out.iter().cloned().collect();
    let n = flat.len() as f32;
    let mean = flat.iter().sum::<f32>() / n;
    let std = (flat.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
    let std = if std < 1e-10 { 1.0 } else { std };

    out.mapv_inplace(|x| (x - mean) / std);
    out
}

/// Transpose from [n_mels, n_frames] to [n_frames, n_mels] as required by CLAP audio model.
/// The model expects [batch, channels, time_frames, n_mels] = [1, 1, 1001, 64].
fn transpose_for_clap(spec: Array2<f32>) -> Array2<f32> {
    spec.t().to_owned()
}

fn top_n_indices(scores: &[f32], n: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(n).map(|(i, _)| *i).collect()
}

/// Compute CLAP-based tags (mood, energy, melody) and combine with
/// pre-computed instrument labels from MTT MusiCNN and genre labels from Discogs-EffNet.
pub fn compute_tags(
    mono_48000: &[f32],
    model_paths: &ModelPaths,
    instruments: Vec<String>,
    genre: Vec<String>,
) -> anyhow::Result<(Tags, Melody)> {
    // Compute and normalize mel spectrogram: [n_mels, n_frames]
    let mel_spec = compute_mel_spectrogram(mono_48000);
    let mel_spec = normalize_frames(mel_spec);
    // Transpose to [n_frames, n_mels] as CLAP expects [batch, 1, time, mels]
    let mel_spec = transpose_for_clap(mel_spec);

    // Reshape to [1, 1, 1001, 64] — [batch, channels, time_frames, n_mels]
    let mel_flat: Vec<f32> = mel_spec.iter().cloned().collect();
    let mel_arr = Array4::<f32>::from_shape_vec((1, 1, TARGET_FRAMES, N_MELS), mel_flat)
        .context("Failed to shape mel tensor")?;

    // Build ONNX session
    let mut session = Session::builder()
        .map_err(|e| anyhow::anyhow!("Failed to create ORT session builder: {}", e))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?
        .with_intra_threads(num_cpus())
        .map_err(|e| anyhow::anyhow!("Failed to set thread count: {}", e))?
        .commit_from_file(&model_paths.audio_encoder)
        .map_err(|e| anyhow::anyhow!("Failed to load audio encoder ONNX: {}", e))?;

    // Create input tensor
    let input_tensor = Tensor::<f32>::from_array(mel_arr)
        .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {}", e))?;

    // Run inference
    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| anyhow::anyhow!("ONNX inference failed: {}", e))?;

    // Extract first output (audio embedding [1, 512])
    let (_shape, embed_slice) = outputs[0_usize]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract embedding: {}", e))?;

    if embed_slice.len() < EMBED_DIM {
        anyhow::bail!(
            "Expected embedding dim >= {}, got {}",
            EMBED_DIM,
            embed_slice.len()
        );
    }
    let embed_slice = &embed_slice[..EMBED_DIM];

    // L2-normalize audio embedding
    let norm = embed_slice
        .iter()
        .map(|&x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(1e-10);
    let audio_embed: Vec<f32> = embed_slice.iter().map(|&x| x / norm).collect();

    // Load pre-computed text embeddings [N_TOTAL, EMBED_DIM]
    let text_bytes =
        std::fs::read(&model_paths.text_embeddings).context("Failed to read text_embeddings.bin")?;

    let expected_bytes = vocab::N_TOTAL * EMBED_DIM * 4;
    if text_bytes.len() != expected_bytes {
        anyhow::bail!(
            "text_embeddings.bin: expected {} bytes ({} labels x {} dims x 4 bytes), got {}",
            expected_bytes,
            vocab::N_TOTAL,
            EMBED_DIM,
            text_bytes.len()
        );
    }

    let text_embeds: Vec<f32> = text_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let text_matrix = Array2::<f32>::from_shape_vec((vocab::N_TOTAL, EMBED_DIM), text_embeds)
        .context("Failed to shape text embedding matrix")?;

    let audio_vec = Array1::<f32>::from_vec(audio_embed);

    // Cosine similarity dot product: [N_TOTAL]
    let scores: Vec<f32> = text_matrix
        .rows()
        .into_iter()
        .map(|row| row.dot(&audio_vec))
        .collect();

    // Slice scores by category (instruments handled by MTT MusiCNN, genre by Discogs-EffNet)
    let mood_scores = &scores[vocab::MOOD_START..vocab::MOOD_START + vocab::N_MOOD];
    let energy_scores = &scores[vocab::ENERGY_START..vocab::ENERGY_START + vocab::N_ENERGY];
    let melody_scores = &scores[vocab::MELODY_START..vocab::MELODY_START + vocab::N_MELODY];

    // Top-N selection
    let mood: Vec<&'static str> = top_n_indices(mood_scores, 3)
        .into_iter()
        .map(|i| vocab::MOOD[i])
        .collect();

    let energy_idx = top_n_indices(energy_scores, 1)[0];
    let energy: &'static str = vocab::ENERGY[energy_idx];

    let melody_descriptors: Vec<&'static str> = top_n_indices(melody_scores, 3)
        .into_iter()
        .map(|i| vocab::MELODY[i])
        .collect();

    Ok((
        Tags {
            instruments,
            mood,
            energy,
            genre,
        },
        Melody {
            descriptors: melody_descriptors,
        },
    ))
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

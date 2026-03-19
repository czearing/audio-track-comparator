/// Listener-engagement quality scoring using Essentia Discogs-EffNet classification heads.
///
/// Pipeline:
///   1. Resample to 16 kHz, compute log-mel patches (same backbone as genre.rs)
///   2. Run discogs_genre.onnx to extract 1280-dim embeddings (output 1)
///   3. Average-pool embeddings across all patches → one [1280] vector
///   4. Feed that vector into each of the three regression head models:
///        engagement.onnx        → model/Identity [1]
///        approachability.onnx   → model/Identity [1]
///        danceability.onnx      → model/Softmax  [2]  (index 0 = danceable prob)
///
/// All head models live in the "quality" subdirectory of the cache dir.
/// The backbone (discogs_genre.onnx) lives in the "genre" subdirectory.
use crate::resample;
use anyhow::Context;
use ndarray::{Array1, Array3};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use serde::Serialize;
use std::path::Path;

const SAMPLE_RATE: u32 = 16_000;
const FRAME_SIZE: usize = 512;
const HOP_SIZE: usize = 256;
const N_MELS: usize = 96;
const PATCH_SIZE: usize = 128;
const N_EMBED: usize = 1280;

const F_MIN: f64 = 0.0;
const F_MAX: f64 = 8_000.0;

#[derive(Serialize, Clone)]
pub struct QualityScores {
    pub engagement: f32,
    pub approachability: f32,
    pub danceability: f32,
    pub hit_potential: f32,
    pub mood_dark_to_happy: f32, // 0=very dark, 1=very happy
    pub mood_aggressive: f32,    // 0=not aggressive, 1=very aggressive
}

// ── mel preprocessing (identical parameters to genre.rs) ─────────────────────

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

// ── helpers ───────────────────────────────────────────────────────────────────

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn build_session(path: &Path) -> anyhow::Result<Session> {
    Session::builder()
        .map_err(|e| anyhow::anyhow!("ORT session builder error: {}", e))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| anyhow::anyhow!("ORT optimization level error: {}", e))?
        .with_intra_threads(num_cpus())
        .map_err(|e| anyhow::anyhow!("ORT thread count error: {}", e))?
        .commit_from_file(path)
        .map_err(|e| anyhow::anyhow!("Failed to load ONNX model {}: {}", path.display(), e))
}

/// Run a regression head model with a single [1280] embedding vector.
/// Returns the single f32 score.
fn run_regression_head(session: &mut Session, embedding: &[f32; N_EMBED]) -> anyhow::Result<f32> {
    let arr = Array1::<f32>::from_vec(embedding.to_vec());
    // Head model expects shape [1, 1280] (batch of 1)
    let arr2d = arr
        .into_shape((1, N_EMBED))
        .map_err(|e| anyhow::anyhow!("Failed to reshape embedding: {}", e))?;
    let tensor = Tensor::<f32>::from_array(arr2d)
        .map_err(|e| anyhow::anyhow!("Failed to create head input tensor: {}", e))?;
    let outputs = session
        .run(ort::inputs![tensor])
        .map_err(|e| anyhow::anyhow!("Head ONNX inference failed: {}", e))?;
    let (_shape, raw) = outputs[0_usize]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract head output: {}", e))?;
    Ok(*raw.iter().next().unwrap_or(&0.0))
}

/// Run the danceability classification head. Output is [2] softmax;
/// index 0 is the "danceable" probability.
fn run_danceability_head(session: &mut Session, embedding: &[f32; N_EMBED]) -> anyhow::Result<f32> {
    let arr = Array1::<f32>::from_vec(embedding.to_vec());
    let arr2d = arr
        .into_shape((1, N_EMBED))
        .map_err(|e| anyhow::anyhow!("Failed to reshape danceability embedding: {}", e))?;
    let tensor = Tensor::<f32>::from_array(arr2d)
        .map_err(|e| anyhow::anyhow!("Failed to create danceability input tensor: {}", e))?;
    let outputs = session
        .run(ort::inputs![tensor])
        .map_err(|e| anyhow::anyhow!("Danceability ONNX inference failed: {}", e))?;
    let (_shape, raw) = outputs[0_usize]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract danceability output: {}", e))?;
    // raw is [danceable_prob, not_danceable_prob]; return danceable probability
    Ok(*raw.iter().next().unwrap_or(&0.0))
}

// ── public API ────────────────────────────────────────────────────────────────

/// Compute quality scores for mono 22050 Hz audio.
///
/// `genre_cache_dir` must contain `discogs_genre.onnx` (backbone).
/// `quality_cache_dir` must contain `engagement.onnx`, `approachability.onnx`,
/// `danceability.onnx`.
/// `emotion_cache_dir` must contain `valence.onnx` (mood_happy) and
/// `arousal.onnx` (mood_relaxed).
pub fn score(
    mono_22050: &[f32],
    genre_cache_dir: &Path,
    quality_cache_dir: &Path,
    emotion_cache_dir: &Path,
) -> anyhow::Result<QualityScores> {
    let backbone_path = genre_cache_dir.join("discogs_genre.onnx");
    let engagement_path = quality_cache_dir.join("engagement.onnx");
    let approachability_path = quality_cache_dir.join("approachability.onnx");
    let danceability_path = quality_cache_dir.join("danceability.onnx");
    let valence_path = emotion_cache_dir.join("valence.onnx");
    let arousal_path = emotion_cache_dir.join("arousal.onnx");

    // Resample 22050 → 16000 Hz
    let mono_16k = resample::resample(mono_22050, 22050, SAMPLE_RATE)
        .context("Failed to resample to 16kHz for quality models")?;

    // Compute mel patches
    let patches = compute_patches(&mono_16k);
    if patches.is_empty() {
        // Not enough audio — return neutral scores
        return Ok(QualityScores {
            engagement: 0.5,
            approachability: 0.5,
            danceability: 0.5,
            hit_potential: 5.0,
            mood_dark_to_happy: 0.5,
            mood_aggressive: 0.5,
        });
    }

    let n_patches = patches.len();

    // Build batch tensor [n_patches, PATCH_SIZE, N_MELS]
    let flat: Vec<f32> = patches.into_iter().flatten().collect();
    let arr = Array3::<f32>::from_shape_vec((n_patches, PATCH_SIZE, N_MELS), flat)
        .context("Failed to shape quality backbone input tensor")?;

    // ── Run backbone to extract embeddings ────────────────────────────────────
    let mut backbone = build_session(&backbone_path)?;

    let input_tensor = Tensor::<f32>::from_array(arr)
        .map_err(|e| anyhow::anyhow!("Failed to create backbone input tensor: {}", e))?;

    let outputs = backbone
        .run(ort::inputs![input_tensor])
        .map_err(|e| anyhow::anyhow!("Backbone ONNX inference failed: {}", e))?;

    // Output 1 is embeddings: [n_patches, 1280]
    let (_shape, embed_raw) = outputs[1_usize]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract backbone embeddings: {}", e))?;

    if embed_raw.len() != n_patches * N_EMBED {
        anyhow::bail!(
            "Backbone embedding size mismatch: expected {}, got {}",
            n_patches * N_EMBED,
            embed_raw.len()
        );
    }

    // Average-pool embeddings across all patches → [1280]
    let mut avg_embed = [0.0f32; N_EMBED];
    for p in 0..n_patches {
        for d in 0..N_EMBED {
            avg_embed[d] += embed_raw[p * N_EMBED + d];
        }
    }
    let n_f = n_patches as f32;
    for v in avg_embed.iter_mut() {
        *v /= n_f;
    }

    // ── Run the five head models ──────────────────────────────────────────────
    let mut eng_session = build_session(&engagement_path)?;
    let mut app_session = build_session(&approachability_path)?;
    let mut dan_session = build_session(&danceability_path)?;
    let mut val_session = build_session(&valence_path)?;
    let mut aro_session = build_session(&arousal_path)?;

    let engagement = run_regression_head(&mut eng_session, &avg_embed)?;
    let approachability = run_regression_head(&mut app_session, &avg_embed)?;
    let danceability = run_danceability_head(&mut dan_session, &avg_embed)?;
    // mood_happy outputs [happy, non_happy]; index 0 = P(happy) = dark_to_happy
    let mood_dark_to_happy = run_danceability_head(&mut val_session, &avg_embed)?;
    // mood_aggressive outputs [aggressive, non_aggressive]; index 0 = P(aggressive)
    let mood_aggressive = run_danceability_head(&mut aro_session, &avg_embed)?;

    // Composite hit potential, rounded to 1 decimal place
    let raw_hit = engagement * 0.45 + approachability * 0.35 + danceability * 0.20;
    let hit_potential = (raw_hit * 10.0 * 10.0).round() / 10.0;

    Ok(QualityScores {
        engagement,
        approachability,
        danceability,
        hit_potential,
        mood_dark_to_happy,
        mood_aggressive,
    })
}

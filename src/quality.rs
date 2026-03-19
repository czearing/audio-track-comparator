/// Listener-engagement quality scoring using Essentia Discogs-EffNet classification heads.
///
/// Pipeline:
///   1. Receive pre-computed BackboneOutput (embeddings already extracted)
///   2. Average-pool embeddings across all patches → one [1280] vector
///   3. Feed that vector into each head model:
///        engagement.onnx        → model/Identity [1]
///        approachability.onnx   → model/Identity [1]
///        danceability.onnx      → model/Softmax  [2]  (index 0 = danceable prob)
///        sad.onnx               → model/Softmax  [2]  (index 1 = sad prob)
///        acoustic.onnx          → model/Softmax  [2]  (index 0 = acoustic prob)
///        timbre.onnx            → model/Softmax  [2]  (index 0 = bright prob)
///
/// All head models live in the "quality" subdirectory of the cache dir.
/// The backbone (discogs_genre.onnx) is run upstream in pipeline.rs.
use crate::backbone::BackboneOutput;
use ndarray::Array1;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use serde::Serialize;
use std::path::Path;

const N_EMBED: usize = 1280;

#[derive(Serialize, Clone)]
pub struct QualityScores {
    pub engagement: f32,
    pub approachability: f32,
    pub danceability: f32,
    pub hit_potential: f32,
    pub mood_dark_to_happy: f32, // 0=very dark, 1=very happy
    pub mood_aggressive: f32,    // 0=not aggressive, 1=very aggressive
    pub mood_sad: f32,           // 0=not sad, 1=melancholic/sad
    pub mood_acoustic: f32,      // 0=electronic, 1=acoustic
    pub timbre_bright: f32,      // 0=dark timbre, 1=bright timbre
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

/// Run a softmax classification head. Output is [2]; returns the value at `positive_index`.
fn run_softmax_head(
    session: &mut Session,
    embedding: &[f32; N_EMBED],
    positive_index: usize,
) -> anyhow::Result<f32> {
    let arr = Array1::<f32>::from_vec(embedding.to_vec());
    let arr2d = arr
        .into_shape((1, N_EMBED))
        .map_err(|e| anyhow::anyhow!("Failed to reshape softmax head embedding: {}", e))?;
    let tensor = Tensor::<f32>::from_array(arr2d)
        .map_err(|e| anyhow::anyhow!("Failed to create softmax head input tensor: {}", e))?;
    let outputs = session
        .run(ort::inputs![tensor])
        .map_err(|e| anyhow::anyhow!("Softmax head ONNX inference failed: {}", e))?;
    let (_shape, raw) = outputs[0_usize]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract softmax head output: {}", e))?;
    let vals: Vec<f32> = raw.iter().cloned().collect();
    Ok(*vals.get(positive_index).unwrap_or(&0.0))
}

/// Run a danceability classification head. Output is [2] softmax;
/// index 0 is the "danceable" probability. Kept for backward compatibility.
fn run_danceability_head(session: &mut Session, embedding: &[f32; N_EMBED]) -> anyhow::Result<f32> {
    run_softmax_head(session, embedding, 0)
}

// ── public API ────────────────────────────────────────────────────────────────

/// Compute quality scores from a pre-computed backbone output.
///
/// `quality_cache_dir` must contain `engagement.onnx`, `approachability.onnx`,
/// `danceability.onnx`, `sad.onnx`, `acoustic.onnx`, `timbre.onnx`.
/// `emotion_cache_dir` must contain `valence.onnx` (mood_happy) and
/// `arousal.onnx` (mood_relaxed).
pub fn score(
    backbone: &BackboneOutput,
    quality_cache_dir: &Path,
    emotion_cache_dir: &Path,
) -> anyhow::Result<QualityScores> {
    let engagement_path = quality_cache_dir.join("engagement.onnx");
    let approachability_path = quality_cache_dir.join("approachability.onnx");
    let danceability_path = quality_cache_dir.join("danceability.onnx");
    let sad_path = quality_cache_dir.join("sad.onnx");
    let acoustic_path = quality_cache_dir.join("acoustic.onnx");
    let timbre_path = quality_cache_dir.join("timbre.onnx");
    let valence_path = emotion_cache_dir.join("valence.onnx");
    let arousal_path = emotion_cache_dir.join("arousal.onnx");

    if backbone.n_patches == 0 || backbone.embeddings.is_empty() {
        // Not enough audio — return neutral scores
        return Ok(QualityScores {
            engagement: 0.5,
            approachability: 0.5,
            danceability: 0.5,
            hit_potential: 5.0,
            mood_dark_to_happy: 0.5,
            mood_aggressive: 0.5,
            mood_sad: 0.5,
            mood_acoustic: 0.5,
            timbre_bright: 0.5,
        });
    }

    let n_patches = backbone.n_patches;
    let embed_raw = &backbone.embeddings;

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

    // ── Run the head models ───────────────────────────────────────────────────
    let mut eng_session = build_session(&engagement_path)?;
    let mut app_session = build_session(&approachability_path)?;
    let mut dan_session = build_session(&danceability_path)?;
    let mut val_session = build_session(&valence_path)?;
    let mut aro_session = build_session(&arousal_path)?;
    let mut sad_session = build_session(&sad_path)?;
    let mut acoustic_session = build_session(&acoustic_path)?;
    let mut timbre_session = build_session(&timbre_path)?;

    let engagement = run_regression_head(&mut eng_session, &avg_embed)?;
    let approachability = run_regression_head(&mut app_session, &avg_embed)?;
    // danceability: [danceable, not_danceable]; index 0 = P(danceable)
    let danceability = run_danceability_head(&mut dan_session, &avg_embed)?;
    // mood_happy: [happy, non_happy]; index 0 = P(happy) = dark_to_happy
    let mood_dark_to_happy = run_danceability_head(&mut val_session, &avg_embed)?;
    // mood_aggressive: [aggressive, non_aggressive]; index 0 = P(aggressive)
    let mood_aggressive = run_danceability_head(&mut aro_session, &avg_embed)?;
    // mood_sad: [non_sad, sad]; index 1 = P(sad)
    let mood_sad = run_softmax_head(&mut sad_session, &avg_embed, 1)?;
    // mood_acoustic: [acoustic, non_acoustic]; index 0 = P(acoustic)
    let mood_acoustic = run_softmax_head(&mut acoustic_session, &avg_embed, 0)?;
    // timbre: [bright, dark]; index 0 = P(bright)
    let timbre_bright = run_softmax_head(&mut timbre_session, &avg_embed, 0)?;

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
        mood_sad,
        mood_acoustic,
        timbre_bright,
    })
}

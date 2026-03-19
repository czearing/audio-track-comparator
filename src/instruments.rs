/// MTG-Jamendo instrument classifier (Discogs-EffNet head).
///
/// Model: instrument_detector.onnx  (mtg_jamendo_instrument-discogs-effnet-1.onnx)
///   Input:  [1, 1280]  float32   (average-pooled Discogs-EffNet embedding)
///   Output: [1, N]     float32   (per-instrument sigmoid probability)
///
/// Uses the same BackboneOutput already computed for genre/quality — no extra
/// resampling or mel-spectrogram pipeline required.
use crate::backbone::BackboneOutput;
use anyhow::Context;
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

const N_EMBED: usize = 1280;
const THRESHOLD: f32 = 0.10;

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Detect instruments from a pre-computed backbone output.
///
/// `cache_dir` must contain `instrument_detector.onnx` and `instrument_labels.json`.
/// Returns labels whose sigmoid probability exceeds THRESHOLD.
pub fn detect(backbone: &BackboneOutput, cache_dir: &Path) -> anyhow::Result<Vec<String>> {
    let onnx_path = cache_dir.join("instrument_detector.onnx");
    let labels_path = cache_dir.join("instrument_labels.json");

    let labels_json =
        std::fs::read_to_string(&labels_path).context("Failed to read instrument_labels.json")?;
    let labels: Vec<String> =
        serde_json::from_str(&labels_json).context("Failed to parse instrument_labels.json")?;
    let n_classes = labels.len();

    if backbone.n_patches == 0 || backbone.embeddings.is_empty() {
        return Ok(Vec::new());
    }

    // Average-pool embeddings across all patches → [1280]
    let n_patches = backbone.n_patches;
    let embed_raw = &backbone.embeddings;
    let mut avg_embed = vec![0.0f32; N_EMBED];
    for p in 0..n_patches {
        for d in 0..N_EMBED {
            avg_embed[d] += embed_raw[p * N_EMBED + d];
        }
    }
    let n_f = n_patches as f32;
    for v in avg_embed.iter_mut() {
        *v /= n_f;
    }

    let arr = Array2::<f32>::from_shape_vec((1, N_EMBED), avg_embed)
        .context("Failed to shape instrument input tensor")?;

    let mut session = Session::builder()
        .map_err(|e| anyhow::anyhow!("ORT session builder error: {}", e))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| anyhow::anyhow!("ORT optimization level error: {}", e))?
        .with_intra_threads(num_cpus())
        .map_err(|e| anyhow::anyhow!("ORT thread count error: {}", e))?
        .commit_from_file(&onnx_path)
        .map_err(|e| anyhow::anyhow!("Failed to load instrument ONNX model: {}", e))?;

    let tensor = Tensor::<f32>::from_array(arr)
        .map_err(|e| anyhow::anyhow!("Failed to create instrument input tensor: {}", e))?;

    let outputs = session
        .run(ort::inputs![tensor])
        .map_err(|e| anyhow::anyhow!("Instrument ONNX inference failed: {}", e))?;

    let (_shape, raw) = outputs[0_usize]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract instrument output tensor: {}", e))?;

    let scores: Vec<f32> = raw.iter().cloned().collect();
    if scores.len() != n_classes {
        anyhow::bail!(
            "Instrument model output size mismatch: expected {}, got {}",
            n_classes,
            scores.len()
        );
    }

    // Collect labels above threshold, sorted by descending score
    let mut hits: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .filter(|(_, &s)| s >= THRESHOLD)
        .map(|(i, &s)| (i, s))
        .collect();
    hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(hits.into_iter().map(|(i, _)| labels[i].clone()).collect())
}

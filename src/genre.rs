/// Discogs-EffNet genre classifier.
///
/// Consumes pre-computed backbone output (genre_probs from BackboneOutput)
/// and returns the top-5 genre label strings.
use crate::backbone::BackboneOutput;
use anyhow::Context;
use std::path::Path;

fn top_n_indices(scores: &[f32], n: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(n).map(|(i, _)| *i).collect()
}

/// Classify top-5 genre tags from a pre-computed backbone output.
///
/// `labels_dir` must contain `genre_labels.json`.
pub fn classify(backbone: &BackboneOutput, labels_dir: &Path) -> anyhow::Result<Vec<String>> {
    let labels_path = labels_dir.join("genre_labels.json");

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

    if backbone.n_patches == 0 || backbone.genre_probs.is_empty() {
        return Ok(Vec::new());
    }

    let n_patches = backbone.n_patches;
    let genre_probs = &backbone.genre_probs;

    if genre_probs.len() != n_patches * 400 {
        anyhow::bail!(
            "genre_probs size mismatch: expected {}, got {}",
            n_patches * 400,
            genre_probs.len()
        );
    }

    // Average predictions across patches
    let mut avg_scores = vec![0.0f32; 400];
    for p in 0..n_patches {
        for g in 0..400 {
            avg_scores[g] += genre_probs[p * 400 + g];
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

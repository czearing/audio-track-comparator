use crate::diff::Diff;
use crate::pipeline::TrackAnalysis;
use crate::quality::QualityScores;
use anyhow::Context;
use serde::Serialize;
use std::path::{Path, PathBuf};

fn strip_unc_prefix(path: &str) -> String {
    path.strip_prefix(r"\\?\").unwrap_or(path).to_owned()
}

pub fn print_summary(reference: &TrackAnalysis, suno: &TrackAnalysis, diff: &Diff) {
    println!();
    println!("=== Audio Track Comparison ===");
    println!();
    println!("                 Reference          Suno");
    println!(
        "  BPM:           {:<18.1} {:.1}",
        reference.bpm_bpm, suno.bpm_bpm
    );
    println!(
        "  BPM source:    {:<18} {}",
        reference.bpm_source, suno.bpm_source
    );
    println!(
        "  Key:           {:<18} {} {}",
        format!("{} {}", reference.key.root, reference.key.mode),
        suno.key.root,
        suno.key.mode
    );
    println!(
        "  Energy:        {:<18} {}",
        reference.tags.energy, suno.tags.energy
    );
    println!(
        "  Top genre:     {:<18} {}",
        reference.tags.genre.first().map(|s| s.as_str()).unwrap_or(""),
        suno.tags.genre.first().map(|s| s.as_str()).unwrap_or("")
    );
    println!(
        "  Hit Potential: {:<18} {}",
        format!("{:.1} / 10", reference.quality.hit_potential),
        format!("{:.1} / 10", suno.quality.hit_potential)
    );
    println!(
        "  Engagement:    {:<18.2} {:.2}",
        reference.quality.engagement, suno.quality.engagement
    );
    println!(
        "  Danceability:  {:<18.2} {:.2}",
        reference.quality.danceability, suno.quality.danceability
    );
    println!(
        "  Dark -> Happy: {:<18.2} {:.2}",
        reference.quality.mood_dark_to_happy, suno.quality.mood_dark_to_happy
    );
    println!(
        "  Aggressive:        {:<15.2} {:.2}",
        reference.quality.mood_aggressive, suno.quality.mood_aggressive
    );
    println!(
        "  Sad:               {:<15.2} {:.2}",
        reference.quality.mood_sad, suno.quality.mood_sad
    );
    println!(
        "  Acoustic:          {:<15.2} {:.2}",
        reference.quality.mood_acoustic, suno.quality.mood_acoustic
    );
    println!(
        "  Timbre Bright:     {:<15.2} {:.2}",
        reference.quality.timbre_bright, suno.quality.timbre_bright
    );
    println!(
        "  Party:             {:<15.2} {:.2}",
        reference.quality.mood_party, suno.quality.mood_party
    );
    println!(
        "  Electronic:        {:<15.2} {:.2}",
        reference.quality.mood_electronic, suno.quality.mood_electronic
    );
    println!();
    println!("  BPM delta:     {:.2}", diff.bpm.delta_bpm);
    println!(
        "  Key match:     {} (distance: {} semitones)",
        diff.key.match_, diff.key.distance_semitones
    );
    println!("  Energy match:  {}", diff.tags.energy_match);
    println!(
        "  Music similarity: {:.1}%",
        diff.similarity_pct
    );
    println!();
}

#[derive(Serialize)]
struct Meta<'a> {
    analyzed_at: &'a str,
    reference_file: String,
    suno_file: String,
}

#[derive(Serialize)]
struct KeyJson<'a> {
    root: &'a str,
    mode: &'a str,
}

#[derive(Serialize)]
struct TagsJson<'a> {
    instruments: &'a [String],
    mood: &'a [&'static str],
    energy: &'static str,
    genre: &'a [String],
}

#[derive(Serialize)]
struct MelodyJson<'a> {
    descriptors: &'a [&'static str],
}

#[derive(Serialize)]
struct TrackJson<'a> {
    bpm_bpm: f32,
    bpm_source: &'a str,
    key: KeyJson<'a>,
    tags: TagsJson<'a>,
    melody: MelodyJson<'a>,
    quality: &'a QualityScores,
}

#[derive(Serialize)]
struct Report<'a> {
    meta: Meta<'a>,
    reference: TrackJson<'a>,
    suno: TrackJson<'a>,
    diff: &'a Diff,
}

pub fn write_report(
    output_dir: Option<&PathBuf>,
    reference: &TrackAnalysis,
    suno: &TrackAnalysis,
    diff: &Diff,
    analyzed_at: &str,
    reference_path: &Path,
    suno_path: &Path,
) -> anyhow::Result<PathBuf> {
    let ref_stem = reference_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("reference");
    let suno_stem = suno_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("suno");

    // Timestamp portion: YYYYMMDD_HHMMSS from analyzed_at (format: YYYY-MM-DDTHH:MM:SSZ)
    let ts = analyzed_at
        .replace('-', "")
        .replace('T', "_")
        .replace(':', "")
        .replace('Z', "");
    let ts = &ts[..15]; // "YYYYMMDD_HHMMSS"

    let filename = format!("{}_vs_{}_{}.json", ref_stem, suno_stem, ts);

    let dir = match output_dir {
        Some(d) => d.clone(),
        None => PathBuf::from("output"),
    };

    std::fs::create_dir_all(&dir)
        .with_context(|| format!("Failed to create output directory: {}", dir.display()))?;

    let report_path = dir.join(&filename);

    let report = Report {
        meta: Meta {
            analyzed_at,
            reference_file: strip_unc_prefix(
                &reference_path
                    .canonicalize()
                    .unwrap_or_else(|_| reference_path.to_path_buf())
                    .to_string_lossy(),
            ),
            suno_file: strip_unc_prefix(
                &suno_path
                    .canonicalize()
                    .unwrap_or_else(|_| suno_path.to_path_buf())
                    .to_string_lossy(),
            ),
        },
        reference: TrackJson {
            bpm_bpm: reference.bpm_bpm,
            bpm_source: reference.bpm_source,
            key: KeyJson {
                root: reference.key.root,
                mode: reference.key.mode,
            },
            tags: TagsJson {
                instruments: &reference.tags.instruments,
                mood: &reference.tags.mood,
                energy: reference.tags.energy,
                genre: &reference.tags.genre,
            },
            melody: MelodyJson {
                descriptors: &reference.melody.descriptors,
            },
            quality: &reference.quality,
        },
        suno: TrackJson {
            bpm_bpm: suno.bpm_bpm,
            bpm_source: suno.bpm_source,
            key: KeyJson {
                root: suno.key.root,
                mode: suno.key.mode,
            },
            tags: TagsJson {
                instruments: &suno.tags.instruments,
                mood: &suno.tags.mood,
                energy: suno.tags.energy,
                genre: &suno.tags.genre,
            },
            melody: MelodyJson {
                descriptors: &suno.melody.descriptors,
            },
            quality: &suno.quality,
        },
        diff,
    };

    let json = serde_json::to_string_pretty(&report).context("Failed to serialize report")?;
    std::fs::write(&report_path, &json)
        .with_context(|| format!("Failed to write report: {}", report_path.display()))?;

    Ok(report_path)
}

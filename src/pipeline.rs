use crate::clap_model::{Melody, Tags};
use crate::decode;
use crate::key::KeyResult;
use crate::model_cache::ModelPaths;
use crate::quality::QualityScores;
use crate::resample;
use std::path::Path;

pub struct TrackAnalysis {
    pub bpm_bpm: f32,
    pub bpm_source: &'static str,
    pub key: KeyResult,
    pub tags: Tags,
    pub melody: Melody,
    pub quality: QualityScores,
}

pub fn analyze_file(path: &Path, model_paths: &ModelPaths) -> anyhow::Result<TrackAnalysis> {
    println!("  Decoding: {}", path.display());
    let buf = decode::decode_mp3(path)?;
    let native_rate = buf.sample_rate;

    let mono = resample::to_mono(&buf);

    println!("  Resampling to 22050 Hz ...");
    let samples_22050 = resample::resample(&mono, native_rate, 22050)?;

    println!("  Resampling to 48000 Hz ...");
    let samples_48000 = resample::resample(&mono, native_rate, 48000)?;

    println!("  Detecting BPM ...");
    let (bpm_bpm, bpm_source) = crate::bpm::detect(&samples_22050);

    println!("  Detecting key ...");
    let key = crate::key::detect(&samples_22050);

    println!("  Running Discogs-EffNet backbone (once) ...");
    let backbone_path = &model_paths.genre_onnx;
    let backbone = crate::backbone::run(&samples_22050, backbone_path)?;

    println!("  Detecting genre (Discogs-EffNet) ...");
    let genre_cache = model_paths.genre_onnx.parent().expect("genre_onnx has no parent");
    let genre = crate::genre::classify(&backbone, genre_cache)?;

    println!("  Detecting instruments (MTG-Jamendo) ...");
    let instrument_cache = model_paths.instrument_onnx.parent().expect("instrument_onnx has no parent");
    let instruments = crate::instruments::detect(&backbone, instrument_cache)?;

    println!("  Running CLAP inference ...");
    let (tags, melody) = crate::clap_model::compute_tags(&samples_48000, model_paths, instruments, genre)?;

    println!("  Scoring quality and emotion (engagement / approachability / danceability / mood) ...");
    let quality_cache = model_paths.quality_dir.as_path();
    let emotion_cache = model_paths.emotion_dir.as_path();
    let quality = crate::quality::score(&backbone, quality_cache, emotion_cache)?;

    Ok(TrackAnalysis {
        bpm_bpm,
        bpm_source,
        key,
        tags,
        melody,
        quality,
    })
}

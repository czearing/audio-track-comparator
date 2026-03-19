use crate::pipeline::TrackAnalysis;
use serde::Serialize;

#[derive(Serialize)]
pub struct QualityDiff {
    pub engagement_delta: f32,
    pub approachability_delta: f32,
    pub danceability_delta: f32,
    pub hit_potential_delta: f32,
    pub mood_dark_to_happy_delta: f32,
    pub mood_aggressive_delta: f32,
    pub mood_sad_delta: f32,
    pub mood_acoustic_delta: f32,
    pub timbre_bright_delta: f32,
}

#[derive(Serialize)]
pub struct BpmDiff {
    pub reference_bpm: f32,
    pub suno_bpm: f32,
    pub delta_bpm: f32,
}

#[derive(Serialize)]
pub struct KeyDiff {
    pub reference_key: String,
    pub suno_key: String,
    #[serde(rename = "match")]
    pub match_: bool,
    pub distance_semitones: u8,
}

#[derive(Serialize)]
pub struct TagsDiff {
    pub instruments_in_reference_not_suno: Vec<String>,
    pub instruments_in_suno_not_reference: Vec<String>,
    pub mood_in_reference_not_suno: Vec<&'static str>,
    pub mood_in_suno_not_reference: Vec<&'static str>,
    pub energy_match: bool,
    pub genre_in_reference_not_suno: Vec<String>,
    pub genre_in_suno_not_reference: Vec<String>,
}

#[derive(Serialize)]
pub struct MelodyDiff {
    pub descriptors_in_reference_not_suno: Vec<&'static str>,
    pub descriptors_in_suno_not_reference: Vec<&'static str>,
}

#[derive(Serialize)]
pub struct Diff {
    pub bpm: BpmDiff,
    pub key: KeyDiff,
    pub tags: TagsDiff,
    pub melody: MelodyDiff,
    pub quality: QualityDiff,
}

fn set_diff_static<'a>(
    a: &[&'a str],
    b: &[&str],
) -> Vec<&'a str> {
    a.iter().filter(|x| !b.contains(x)).copied().collect()
}

fn set_diff_owned(a: &[String], b: &[String]) -> Vec<String> {
    a.iter().filter(|x| !b.contains(x)).cloned().collect()
}

fn pitch_class_index(root: &str) -> u8 {
    match root {
        "C" => 0,
        "C#" => 1,
        "D" => 2,
        "D#" => 3,
        "E" => 4,
        "F" => 5,
        "F#" => 6,
        "G" => 7,
        "G#" => 8,
        "A" => 9,
        "A#" => 10,
        "B" => 11,
        _ => 0,
    }
}

pub fn compute(reference: &TrackAnalysis, suno: &TrackAnalysis) -> Diff {
    // BPM diff — round delta to 2 decimal places
    let delta_bpm = ((suno.bpm_bpm - reference.bpm_bpm) as f64 * 100.0).round() as f32 / 100.0;

    // Key diff
    let ref_key = format!("{} {}", reference.key.root, reference.key.mode);
    let suno_key = format!("{} {}", suno.key.root, suno.key.mode);
    let key_match = ref_key == suno_key;

    let ref_pc = pitch_class_index(reference.key.root);
    let suno_pc = pitch_class_index(suno.key.root);
    let diff_abs = (ref_pc as i16 - suno_pc as i16).unsigned_abs() as u8;
    let distance_semitones = diff_abs.min(12 - diff_abs);

    // Quality deltas — round to 2 decimal places
    let round2 = |v: f32| (v * 100.0).round() / 100.0;
    let engagement_delta = round2(suno.quality.engagement - reference.quality.engagement);
    let approachability_delta =
        round2(suno.quality.approachability - reference.quality.approachability);
    let danceability_delta = round2(suno.quality.danceability - reference.quality.danceability);
    let hit_potential_delta = round2(suno.quality.hit_potential - reference.quality.hit_potential);
    let mood_dark_to_happy_delta =
        round2(suno.quality.mood_dark_to_happy - reference.quality.mood_dark_to_happy);
    let mood_aggressive_delta =
        round2(suno.quality.mood_aggressive - reference.quality.mood_aggressive);
    let mood_sad_delta = round2(suno.quality.mood_sad - reference.quality.mood_sad);
    let mood_acoustic_delta = round2(suno.quality.mood_acoustic - reference.quality.mood_acoustic);
    let timbre_bright_delta = round2(suno.quality.timbre_bright - reference.quality.timbre_bright);

    Diff {
        bpm: BpmDiff {
            reference_bpm: reference.bpm_bpm,
            suno_bpm: suno.bpm_bpm,
            delta_bpm,
        },
        key: KeyDiff {
            reference_key: ref_key,
            suno_key,
            match_: key_match,
            distance_semitones,
        },
        tags: TagsDiff {
            instruments_in_reference_not_suno: set_diff_owned(
                &reference.tags.instruments,
                &suno.tags.instruments,
            ),
            instruments_in_suno_not_reference: set_diff_owned(
                &suno.tags.instruments,
                &reference.tags.instruments,
            ),
            mood_in_reference_not_suno: set_diff_static(
                &reference.tags.mood,
                &suno.tags.mood,
            ),
            mood_in_suno_not_reference: set_diff_static(
                &suno.tags.mood,
                &reference.tags.mood,
            ),
            energy_match: reference.tags.energy == suno.tags.energy,
            genre_in_reference_not_suno: set_diff_owned(
                &reference.tags.genre,
                &suno.tags.genre,
            ),
            genre_in_suno_not_reference: set_diff_owned(
                &suno.tags.genre,
                &reference.tags.genre,
            ),
        },
        melody: MelodyDiff {
            descriptors_in_reference_not_suno: set_diff_static(
                &reference.melody.descriptors,
                &suno.melody.descriptors,
            ),
            descriptors_in_suno_not_reference: set_diff_static(
                &suno.melody.descriptors,
                &reference.melody.descriptors,
            ),
        },
        quality: QualityDiff {
            engagement_delta,
            approachability_delta,
            danceability_delta,
            hit_potential_delta,
            mood_dark_to_happy_delta,
            mood_aggressive_delta,
            mood_sad_delta,
            mood_acoustic_delta,
            timbre_bright_delta,
        },
    }
}

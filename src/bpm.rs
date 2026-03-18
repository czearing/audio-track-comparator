use aubio_rs::{OnsetMode, Tempo};

pub const BPM_FALLBACK: f32 = 120.0;
pub const BPM_MIN: f32 = 40.0;
pub const BPM_MAX: f32 = 240.0;

pub fn detect(mono_22050: &[f32]) -> (f32, &'static str) {
    // aubio Tempo requires a hop size and window size
    // Standard values: buf_size=1024, hop_size=512 at 22050 Hz
    let buf_size: usize = 1024;
    let hop_size: usize = 512;
    let sample_rate: u32 = 22050;

    if mono_22050.len() < buf_size {
        return (BPM_FALLBACK, "fallback");
    }

    let mut tempo = match Tempo::new(
        OnsetMode::SpecDiff,
        buf_size,
        hop_size,
        sample_rate,
    ) {
        Ok(t) => t,
        Err(_) => return (BPM_FALLBACK, "fallback"),
    };

    // Feed audio in hop_size chunks
    for chunk in mono_22050.chunks(hop_size) {
        if chunk.len() < hop_size {
            break; // skip incomplete final chunk
        }
        let _ = tempo.do_result(chunk);
    }

    let bpm = tempo.get_bpm();

    if bpm < BPM_MIN || bpm > BPM_MAX {
        return (BPM_FALLBACK, "fallback");
    }

    // Harmonic normalization: subharmonics (half-tempo) and superharmonics
    // (double-tempo) are common beat tracking errors. Normalize into [90, 180]
    // — the range where produced music almost always lives — by doubling or
    // halving until we land in range. This corrects the classic "dubstep
    // half-time" detection error where aubio locks onto 63 instead of 126.
    let normalized = normalize_tempo(bpm);
    (normalized, "detected")
}

fn normalize_tempo(mut bpm: f32) -> f32 {
    // Prefer the [90, 180] window. Repeatedly double or halve until inside it,
    // or until we'd overshoot in the wrong direction.
    while bpm < 90.0 && bpm * 2.0 <= BPM_MAX {
        bpm *= 2.0;
    }
    while bpm > 180.0 && bpm / 2.0 >= BPM_MIN {
        bpm /= 2.0;
    }
    (bpm * 100.0).round() / 100.0
}

use rustfft::{num_complex::Complex, FftPlanner};

pub struct KeyResult {
    pub root: &'static str,
    pub mode: &'static str,
}

const PITCH_NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

const KS_MAJOR: [f32; 12] = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KS_MINOR: [f32; 12] = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

const SAMPLE_RATE: f32 = 22050.0;
const FRAME_SIZE: usize = 4096;
const HOP_SIZE: usize = 2048;

pub fn detect(mono_22050: &[f32]) -> KeyResult {
    if mono_22050.len() < FRAME_SIZE {
        return KeyResult { root: "C", mode: "major" };
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FRAME_SIZE);

    let window: Vec<f32> = (0..FRAME_SIZE)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (FRAME_SIZE - 1) as f32).cos()))
        .collect();

    let n_frames = (mono_22050.len() - FRAME_SIZE) / HOP_SIZE + 1;
    let mut chroma = [0.0f32; 12];
    let mut buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); FRAME_SIZE];

    for frame_idx in 0..n_frames {
        let start = frame_idx * HOP_SIZE;

        for i in 0..FRAME_SIZE {
            let s = if start + i < mono_22050.len() {
                mono_22050[start + i]
            } else {
                0.0
            };
            buf[i] = Complex::new(s * window[i], 0.0);
        }

        fft.process(&mut buf);

        let n_bins = FRAME_SIZE / 2 + 1;
        for k in 1..n_bins {
            let freq = k as f32 * SAMPLE_RATE / FRAME_SIZE as f32;
            if freq < 27.5 || freq > 4186.0 {
                continue;
            }
            let mag = buf[k].norm();
            // Map frequency to pitch class
            let pitch_f = 12.0 * (freq / 440.0).log2();
            let pitch_class = (pitch_f.round() as i32).rem_euclid(12) as usize;
            chroma[pitch_class] += mag;
        }
    }

    let sum: f32 = chroma.iter().sum();
    if sum < 1e-10 {
        return KeyResult { root: "C", mode: "major" };
    }
    let chroma: Vec<f32> = chroma.iter().map(|&x| x / sum).collect();

    // Try all 24 rotations (12 major + 12 minor) via Pearson correlation
    let mut best_score = f32::NEG_INFINITY;
    let mut best_root = 0usize;
    let mut best_mode = "major";

    let ks_mean_major: f32 = KS_MAJOR.iter().sum::<f32>() / 12.0;
    let ks_mean_minor: f32 = KS_MINOR.iter().sum::<f32>() / 12.0;
    let chroma_mean: f32 = chroma.iter().sum::<f32>() / 12.0;

    let chroma_std: f32 = (chroma.iter().map(|&x| (x - chroma_mean).powi(2)).sum::<f32>() / 12.0).sqrt();
    if chroma_std < 1e-10 {
        return KeyResult { root: "C", mode: "major" };
    }

    for root in 0..12 {
        for (mode_idx, (profile, mean)) in [
            (KS_MAJOR.as_ref(), ks_mean_major),
            (KS_MINOR.as_ref(), ks_mean_minor),
        ]
        .iter()
        .enumerate()
        {
            let profile_std: f32 = (profile.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / 12.0).sqrt();
            if profile_std < 1e-10 {
                continue;
            }

            let mut cov = 0.0f32;
            for i in 0..12 {
                let rotated_i = (i + root) % 12;
                cov += (chroma[rotated_i] - chroma_mean) * (profile[i] - mean);
            }
            let r = cov / (12.0 * chroma_std * profile_std);

            if r > best_score {
                best_score = r;
                best_root = root;
                best_mode = if mode_idx == 0 { "major" } else { "minor" };
            }
        }
    }

    KeyResult {
        root: PITCH_NAMES[best_root],
        mode: best_mode,
    }
}

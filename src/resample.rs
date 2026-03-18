use crate::decode::PcmBuffer;
use anyhow::Context;
use rubato::{FftFixedIn, Resampler};

pub fn to_mono(buf: &PcmBuffer) -> Vec<f32> {
    let ch = buf.channels as usize;
    if ch == 1 {
        return buf.samples.clone();
    }
    let n_frames = buf.samples.len() / ch;
    let mut mono = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let mut sum = 0.0f32;
        for c in 0..ch {
            sum += buf.samples[i * ch + c];
        }
        mono.push(sum / ch as f32);
    }
    mono
}

pub fn resample(mono: &[f32], from_rate: u32, to_rate: u32) -> anyhow::Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(mono.to_vec());
    }

    let chunk_size = 1024usize;
    let mut resampler = FftFixedIn::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        chunk_size,
        2,
        1,
    )
    .context("Failed to create resampler")?;

    let mut output: Vec<f32> = Vec::new();
    let mut pos = 0usize;

    while pos + chunk_size <= mono.len() {
        let chunk = vec![mono[pos..pos + chunk_size].to_vec()];
        let out = resampler.process(&chunk, None).context("Resample failed")?;
        output.extend_from_slice(&out[0]);
        pos += chunk_size;
    }

    // Handle remaining samples by zero-padding
    if pos < mono.len() {
        let mut last_chunk = mono[pos..].to_vec();
        last_chunk.resize(chunk_size, 0.0);
        let chunk = vec![last_chunk];
        let out = resampler.process(&chunk, None).context("Resample tail failed")?;
        // Only keep the proportional number of output samples
        let remaining = mono.len() - pos;
        let expected_out = (remaining as f64 * to_rate as f64 / from_rate as f64).ceil() as usize;
        let take = expected_out.min(out[0].len());
        output.extend_from_slice(&out[0][..take]);
    }

    Ok(output)
}

use anyhow::{bail, Context};
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub struct PcmBuffer {
    pub samples: Vec<f32>,
    pub channels: u32,
    pub sample_rate: u32,
}

pub fn decode_mp3(path: &Path) -> anyhow::Result<PcmBuffer> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open: {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("mp3");

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .context("Failed to probe MP3 format")?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .context("No supported audio track found")?;

    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .context("No sample rate in codec params")?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u32)
        .unwrap_or(2);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Failed to create decoder")?;

    let mut samples: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => {
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };

        let spec = *decoded.spec();
        let duration = decoded.capacity() as u64;

        let sb = sample_buf.get_or_insert_with(|| SampleBuffer::new(duration, spec));
        sb.copy_interleaved_ref(decoded);
        samples.extend_from_slice(sb.samples());
    }

    if samples.is_empty() {
        bail!("Decoded zero samples from {}", path.display());
    }

    Ok(PcmBuffer {
        samples,
        channels,
        sample_rate,
    })
}

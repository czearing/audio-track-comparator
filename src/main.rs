mod bpm;
mod clap_model;
mod decode;
mod diff;
mod genre;
mod instruments;
mod key;
mod model_cache;
mod output;
mod pipeline;
mod quality;
mod resample;
mod vocab;

use chrono::Utc;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "audio-track-comparator")]
#[command(about = "Compare a reference MP3 and a Suno-generated MP3, output structured JSON diff")]
struct Cli {
    #[arg(long, help = "Path to the reference MP3 file")]
    reference: PathBuf,

    #[arg(long, help = "Path to the Suno-generated MP3 file")]
    suno: PathBuf,

    #[arg(long, help = "Output directory for the JSON report (default: output/)")]
    output: Option<PathBuf>,

    #[arg(
        long,
        help = "Exit with error if CLAP model is not already cached (no download)"
    )]
    no_cache_download: bool,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Validate reference file
    if !cli.reference.exists() {
        eprintln!(
            "ERROR: Reference file not found: {}",
            cli.reference.display()
        );
        std::process::exit(1);
    }
    if cli.reference.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).as_deref() != Some("mp3") {
        eprintln!(
            "ERROR: Reference file must have .mp3 extension: {}",
            cli.reference.display()
        );
        std::process::exit(1);
    }

    // Validate suno file
    if !cli.suno.exists() {
        eprintln!("ERROR: Suno file not found: {}", cli.suno.display());
        std::process::exit(1);
    }
    if cli.suno.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).as_deref() != Some("mp3") {
        eprintln!(
            "ERROR: Suno file must have .mp3 extension: {}",
            cli.suno.display()
        );
        std::process::exit(1);
    }

    // Ensure CLAP model is cached (exits with code 1 if not)
    let model_paths = model_cache::ensure_model(cli.no_cache_download)?;

    // Record start time
    let analyzed_at = Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

    // Analyze reference track
    println!("\nAnalyzing reference: {}", cli.reference.display());
    let ref_analysis = pipeline::analyze_file(&cli.reference, &model_paths)?;

    // Analyze suno track
    println!("\nAnalyzing suno: {}", cli.suno.display());
    let suno_analysis = pipeline::analyze_file(&cli.suno, &model_paths)?;

    // Compute diff
    let diff = diff::compute(&ref_analysis, &suno_analysis);

    // Print summary before file write (AC-37)
    output::print_summary(&ref_analysis, &suno_analysis, &diff);

    // Write JSON report
    let report_path = output::write_report(
        cli.output.as_ref(),
        &ref_analysis,
        &suno_analysis,
        &diff,
        &analyzed_at,
        &cli.reference,
        &cli.suno,
    )?;

    println!("Report written to: {}", report_path.display());

    Ok(())
}

use std::path::PathBuf;

pub struct ModelPaths {
    pub audio_encoder: PathBuf,
    pub text_embeddings: PathBuf,
    pub genre_onnx: PathBuf,
    pub instrument_onnx: PathBuf,
}

fn cache_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Cannot determine home directory")
        .join(".cache")
        .join("audio-track-comparator")
        .join("clap")
}

fn genre_cache_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Cannot determine home directory")
        .join(".cache")
        .join("audio-track-comparator")
        .join("genre")
}

fn instrument_cache_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Cannot determine home directory")
        .join(".cache")
        .join("audio-track-comparator")
        .join("instruments")
}

/// Locate an export script relative to the running binary, falling back to cwd.
fn find_script(name: &str) -> Option<PathBuf> {
    // 1. Next to the binary
    if let Ok(exe) = std::env::current_exe() {
        let candidate = exe
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("scripts")
            .join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    // 2. Relative to cwd
    let candidate = PathBuf::from("scripts").join(name);
    if candidate.exists() {
        return Some(candidate);
    }
    None
}

fn find_python() -> Option<&'static str> {
    for candidate in &["python", "python3"] {
        if std::process::Command::new(candidate)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok()
        {
            return Some(candidate);
        }
    }
    None
}

fn run_script(python: &str, script: &std::path::Path, description: &str) {
    println!("Running: {} {}", python, script.display());
    println!();
    let status = std::process::Command::new(python)
        .arg(script)
        .status()
        .unwrap_or_else(|e| {
            eprintln!("ERROR: Failed to spawn {}: {}", description, e);
            std::process::exit(1);
        });
    if !status.success() {
        eprintln!();
        eprintln!(
            "ERROR: {} script failed (exit code: {:?}).",
            description,
            status.code()
        );
        std::process::exit(1);
    }
}

pub fn ensure_model(no_cache_download: bool) -> anyhow::Result<ModelPaths> {
    let clap_dir = cache_dir();
    let audio_encoder = clap_dir.join("clap-htsat-unfused-audio.onnx");
    let text_embeddings = clap_dir.join("text_embeddings.bin");

    let genre_dir = genre_cache_dir();
    let genre_onnx = genre_dir.join("discogs_genre.onnx");
    let genre_labels = genre_dir.join("genre_labels.json");

    let instrument_dir = instrument_cache_dir();
    let instrument_onnx = instrument_dir.join("instrument_detector.onnx");
    let instrument_labels = instrument_dir.join("instrument_labels.json");

    let clap_present = audio_encoder.exists() && text_embeddings.exists();
    let genre_present = genre_onnx.exists() && genre_labels.exists();
    let instrument_present = instrument_onnx.exists() && instrument_labels.exists();

    if clap_present && genre_present && instrument_present {
        println!("CLAP model cache:       {}", clap_dir.display());
        println!("Genre model cache:      {}", genre_dir.display());
        println!("Instrument model cache: {}", instrument_dir.display());
        return Ok(ModelPaths {
            audio_encoder,
            text_embeddings,
            genre_onnx,
            instrument_onnx,
        });
    }

    if no_cache_download {
        eprintln!("ERROR: Model files not found in cache.");
        eprintln!();
        eprintln!("The --no-cache-download flag prevents downloading.");
        eprintln!("Remove --no-cache-download and re-run to download automatically.");
        eprintln!();
        eprintln!("Missing files:");
        if !audio_encoder.exists() {
            eprintln!("  {}", audio_encoder.display());
        }
        if !text_embeddings.exists() {
            eprintln!("  {}", text_embeddings.display());
        }
        if !genre_onnx.exists() {
            eprintln!("  {}", genre_onnx.display());
        }
        if !genre_labels.exists() {
            eprintln!("  {}", genre_labels.display());
        }
        if !instrument_onnx.exists() {
            eprintln!("  {}", instrument_onnx.display());
        }
        if !instrument_labels.exists() {
            eprintln!("  {}", instrument_labels.display());
        }
        std::process::exit(1);
    }

    let python = match find_python() {
        Some(p) => p,
        None => {
            eprintln!("ERROR: Python is not available on this system.");
            eprintln!("Install Python 3 and re-run.");
            std::process::exit(1);
        }
    };

    // Download CLAP model if needed
    if !clap_present {
        println!("Downloading CLAP model from HuggingFace...");
        println!("(One-time setup. Model will be cached for future runs.)");
        println!();

        let script = match find_script("export_clap_onnx.py") {
            Some(p) => p,
            None => {
                eprintln!("ERROR: Could not find scripts/export_clap_onnx.py.");
                eprintln!("Ensure you are running from the project root directory.");
                std::process::exit(1);
            }
        };

        run_script(python, &script, "CLAP export");

        if !audio_encoder.exists() || !text_embeddings.exists() {
            eprintln!("ERROR: CLAP export completed but expected files are still missing.");
            if !audio_encoder.exists() {
                eprintln!("  {}", audio_encoder.display());
            }
            if !text_embeddings.exists() {
                eprintln!("  {}", text_embeddings.display());
            }
            std::process::exit(1);
        }
        println!();
        println!("CLAP model downloaded successfully.");
    }

    // Download genre model if needed
    if !genre_present {
        println!("Downloading Discogs genre model...");
        println!("(One-time setup. Model will be cached for future runs.)");
        println!();

        let script = match find_script("export_genre_model.py") {
            Some(p) => p,
            None => {
                eprintln!("ERROR: Could not find scripts/export_genre_model.py.");
                eprintln!("Ensure you are running from the project root directory.");
                std::process::exit(1);
            }
        };

        run_script(python, &script, "genre model export");

        if !genre_onnx.exists() || !genre_labels.exists() {
            eprintln!("ERROR: Genre export completed but expected files are still missing.");
            if !genre_onnx.exists() {
                eprintln!("  {}", genre_onnx.display());
            }
            if !genre_labels.exists() {
                eprintln!("  {}", genre_labels.display());
            }
            std::process::exit(1);
        }
        println!();
        println!("Genre model downloaded successfully.");
    }

    // Download instrument model if needed
    if !instrument_present {
        println!("Downloading MTT MusiCNN instrument model...");
        println!("(One-time setup. Model will be cached for future runs.)");
        println!();

        let script = match find_script("export_instrument_model.py") {
            Some(p) => p,
            None => {
                eprintln!("ERROR: Could not find scripts/export_instrument_model.py.");
                eprintln!("Ensure you are running from the project root directory.");
                std::process::exit(1);
            }
        };

        run_script(python, &script, "instrument model export");

        if !instrument_onnx.exists() || !instrument_labels.exists() {
            eprintln!("ERROR: Instrument export completed but expected files are still missing.");
            if !instrument_onnx.exists() {
                eprintln!("  {}", instrument_onnx.display());
            }
            if !instrument_labels.exists() {
                eprintln!("  {}", instrument_labels.display());
            }
            std::process::exit(1);
        }
        println!();
        println!("Instrument model downloaded successfully.");
    }

    println!("CLAP model cache:       {}", clap_dir.display());
    println!("Genre model cache:      {}", genre_dir.display());
    println!("Instrument model cache: {}", instrument_dir.display());

    Ok(ModelPaths {
        audio_encoder,
        text_embeddings,
        genre_onnx,
        instrument_onnx,
    })
}

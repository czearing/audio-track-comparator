use std::path::PathBuf;

pub struct ModelPaths {
    pub audio_encoder: PathBuf,
    pub text_embeddings: PathBuf,
    pub genre_onnx: PathBuf,
    pub instrument_onnx: PathBuf,
    pub quality_dir: PathBuf,
    pub emotion_dir: PathBuf,
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

fn quality_cache_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Cannot determine home directory")
        .join(".cache")
        .join("audio-track-comparator")
        .join("quality")
}

fn emotion_cache_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Cannot determine home directory")
        .join(".cache")
        .join("audio-track-comparator")
        .join("emotion")
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

    let quality_dir = quality_cache_dir();
    let quality_engagement = quality_dir.join("engagement.onnx");
    let quality_approachability = quality_dir.join("approachability.onnx");
    let quality_danceability = quality_dir.join("danceability.onnx");
    let quality_sad = quality_dir.join("sad.onnx");
    let quality_acoustic = quality_dir.join("acoustic.onnx");
    let quality_timbre = quality_dir.join("timbre.onnx");

    let emotion_dir = emotion_cache_dir();
    let emotion_valence = emotion_dir.join("valence.onnx");
    let emotion_arousal = emotion_dir.join("arousal.onnx");

    let clap_present = audio_encoder.exists() && text_embeddings.exists();
    let genre_present = genre_onnx.exists() && genre_labels.exists();
    let instrument_present = instrument_onnx.exists() && instrument_labels.exists();
    let quality_present =
        quality_engagement.exists() && quality_approachability.exists() && quality_danceability.exists()
        && quality_sad.exists() && quality_acoustic.exists() && quality_timbre.exists();
    let emotion_present = emotion_valence.exists() && emotion_arousal.exists();

    if clap_present && genre_present && instrument_present && quality_present && emotion_present {
        println!("CLAP model cache:       {}", clap_dir.display());
        println!("Genre model cache:      {}", genre_dir.display());
        println!("Instrument model cache: {}", instrument_dir.display());
        println!("Quality model cache:    {}", quality_dir.display());
        println!("Emotion model cache:    {}", emotion_dir.display());
        return Ok(ModelPaths {
            audio_encoder,
            text_embeddings,
            genre_onnx,
            instrument_onnx,
            quality_dir,
            emotion_dir,
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
        if !quality_engagement.exists() {
            eprintln!("  {}", quality_engagement.display());
        }
        if !quality_approachability.exists() {
            eprintln!("  {}", quality_approachability.display());
        }
        if !quality_danceability.exists() {
            eprintln!("  {}", quality_danceability.display());
        }
        if !quality_sad.exists() {
            eprintln!("  {}", quality_sad.display());
        }
        if !quality_acoustic.exists() {
            eprintln!("  {}", quality_acoustic.display());
        }
        if !quality_timbre.exists() {
            eprintln!("  {}", quality_timbre.display());
        }
        if !emotion_valence.exists() {
            eprintln!("  {}", emotion_valence.display());
        }
        if !emotion_arousal.exists() {
            eprintln!("  {}", emotion_arousal.display());
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

    // Download quality models if needed
    if !quality_present {
        println!("Downloading Essentia quality models (engagement / approachability / danceability)...");
        println!("(One-time setup. Models will be cached for future runs.)");
        println!();

        let script = match find_script("export_quality_models.py") {
            Some(p) => p,
            None => {
                eprintln!("ERROR: Could not find scripts/export_quality_models.py.");
                eprintln!("Ensure you are running from the project root directory.");
                std::process::exit(1);
            }
        };

        run_script(python, &script, "quality models export");

        if !quality_engagement.exists() || !quality_approachability.exists() || !quality_danceability.exists()
            || !quality_sad.exists() || !quality_acoustic.exists() || !quality_timbre.exists()
        {
            eprintln!("ERROR: Quality export completed but expected files are still missing.");
            if !quality_engagement.exists() {
                eprintln!("  {}", quality_engagement.display());
            }
            if !quality_approachability.exists() {
                eprintln!("  {}", quality_approachability.display());
            }
            if !quality_danceability.exists() {
                eprintln!("  {}", quality_danceability.display());
            }
            if !quality_sad.exists() {
                eprintln!("  {}", quality_sad.display());
            }
            if !quality_acoustic.exists() {
                eprintln!("  {}", quality_acoustic.display());
            }
            if !quality_timbre.exists() {
                eprintln!("  {}", quality_timbre.display());
            }
            std::process::exit(1);
        }
        println!();
        println!("Quality models downloaded successfully.");
    }

    // Download emotion models if needed
    if !emotion_present {
        println!("Downloading Essentia emotion models (mood_happy / mood_relaxed)...");
        println!("(One-time setup. Models will be cached for future runs.)");
        println!();

        let script = match find_script("export_emotion_models.py") {
            Some(p) => p,
            None => {
                eprintln!("ERROR: Could not find scripts/export_emotion_models.py.");
                eprintln!("Ensure you are running from the project root directory.");
                std::process::exit(1);
            }
        };

        run_script(python, &script, "emotion models export");

        if !emotion_valence.exists() || !emotion_arousal.exists() {
            eprintln!("ERROR: Emotion export completed but expected files are still missing.");
            if !emotion_valence.exists() {
                eprintln!("  {}", emotion_valence.display());
            }
            if !emotion_arousal.exists() {
                eprintln!("  {}", emotion_arousal.display());
            }
            std::process::exit(1);
        }
        println!();
        println!("Emotion models downloaded successfully.");
    }

    println!("CLAP model cache:       {}", clap_dir.display());
    println!("Genre model cache:      {}", genre_dir.display());
    println!("Instrument model cache: {}", instrument_dir.display());
    println!("Quality model cache:    {}", quality_dir.display());
    println!("Emotion model cache:    {}", emotion_dir.display());

    Ok(ModelPaths {
        audio_encoder,
        text_embeddings,
        genre_onnx,
        instrument_onnx,
        quality_dir,
        emotion_dir,
    })
}

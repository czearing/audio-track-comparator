// INSTRUMENTS vocab removed — instrument detection now uses the MTT MusiCNN model
// (see src/instruments.rs). N_INSTRUMENTS is kept to preserve the
// text_embeddings.bin layout used by CLAP for mood/energy/melody scoring.

pub const MOOD: &[&str] = &[
    "dark",
    "bright",
    "melancholic",
    "euphoric",
    "aggressive",
    "chill",
    "romantic",
    "haunting",
    "uplifting",
    "mysterious",
    "nostalgic",
    "tense",
    "playful",
    "dreamy",
    "raw",
    "emotional",
    "epic",
];

pub const ENERGY: &[&str] = &["low energy", "medium energy", "high energy", "intense"];

// GENRE vocab removed — genre detection now uses the Discogs-EffNet model
// (see src/genre.rs). N_GENRE and GENRE_START are kept to preserve the
// text_embeddings.bin layout used by CLAP for instruments/mood/energy/melody.

pub const MELODY: &[&str] = &[
    "repetitive",
    "melodic",
    "complex",
    "simple",
    "driving",
    "catchy",
    "atonal",
    "chromatic",
    "pentatonic",
    "modal",
    "anthemic",
    "hypnotic",
    "sparse",
    "layered",
    "evolving",
];

pub const N_INSTRUMENTS: usize = 30;
pub const N_MOOD: usize = 17;
pub const N_ENERGY: usize = 4;
pub const N_GENRE: usize = 91;
pub const N_MELODY: usize = 15;
pub const N_TOTAL: usize = 157;

// Index ranges for each category in the flat embedding array
// [0..30] instruments
// [30..47] mood
// [47..51] energy
// [51..83] genre
// [83..98] melody
pub const MOOD_START: usize = N_INSTRUMENTS;
pub const ENERGY_START: usize = N_INSTRUMENTS + N_MOOD;
#[allow(dead_code)]
pub const GENRE_START: usize = N_INSTRUMENTS + N_MOOD + N_ENERGY;
pub const MELODY_START: usize = N_INSTRUMENTS + N_MOOD + N_ENERGY + N_GENRE;

//! Binary genome dump: persist intermediate results on abort, resume on next run.
use crate::{Gene, Genome};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::File,
    io::{self, BufReader, BufWriter},
    path::Path,
};

/// Write raw genomes to a binary file (bincode). Overwrites existing file.
pub fn save_dump<G: Gene + Serialize>(path: &Path, genomes: &[Genome<G>]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    bincode::serde::encode_into_std_write(genomes, &mut writer, bincode::config::standard())
        .map_err(io::Error::other)?;
    Ok(())
}

/// Read genomes from a binary dump.
/// Returns `None` when the file is missing, corrupted,
/// or genome lengths don't match `genome_len` (stale dump from another config).
pub fn load_dump<G: Gene + DeserializeOwned>(
    path: &Path,
    genome_len: usize,
) -> Option<Vec<Genome<G>>> {
    let mut reader = BufReader::new(File::open(path).ok()?);
    let genomes: Vec<Genome<G>> =
        bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())
            .map_err(|e| eprintln!("Failed to decode dump {}: {e}", path.display()))
            .ok()?;

    if genomes.is_empty() || genomes.iter().any(|g| g.len() != genome_len) {
        eprintln!(
            "Ignoring stale dump {}: genome length mismatch",
            path.display()
        );
        return None;
    }

    Some(genomes)
}

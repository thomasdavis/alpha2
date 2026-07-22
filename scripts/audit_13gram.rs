//! Streaming exact 13-gram contamination audit for Alpha's frozen eval.
//!
//! Build and run without Cargo dependencies:
//!
//!   rustc +1.88 -O scripts/audit_13gram.rs -o /tmp/alpha-audit-13gram
//!   nice -n19 ionice -c3 /tmp/alpha-audit-13gram \
//!     EVAL_DOCS.txt REPORT.tsv TRAIN.txt [TRAIN2.txt ...]
//!
//! "Word" is intentionally fixed and reproducible: a maximal ASCII
//! alphanumeric byte sequence, lower-cased. A gram is the ordered sequence of
//! 13 such words. Two independent rolling u64 hashes are packed into u128;
//! the collision probability is negligible for this audit. Document state is
//! reset at Alpha's <|end_of_text|> marker, so grams never cross boundaries.

use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

const N: usize = 13;
const START: &[u8] = b"@@ALPHA_EVAL_DOC\t";
const END: &[u8] = b"@@END_ALPHA_EVAL_DOC";
const EOT: &[u8] = b"<|end_of_text|>";
const BASE1: u64 = 0x9e3779b185ebca87;
const BASE2: u64 = 0xc2b2ae3d27d4eb4f;

#[derive(Default)]
struct RollingGram {
    words: VecDeque<(u64, u64)>,
    hash1: u64,
    hash2: u64,
    pow1: u64,
    pow2: u64,
}

impl RollingGram {
    fn new() -> Self {
        let mut pow1 = 1u64;
        let mut pow2 = 1u64;
        for _ in 0..N {
            pow1 = pow1.wrapping_mul(BASE1);
            pow2 = pow2.wrapping_mul(BASE2);
        }
        Self {
            words: VecDeque::with_capacity(N),
            hash1: 0,
            hash2: 0,
            pow1,
            pow2,
        }
    }

    fn clear(&mut self) {
        self.words.clear();
        self.hash1 = 0;
        self.hash2 = 0;
    }

    fn push(&mut self, word: (u64, u64)) -> Option<u128> {
        self.hash1 = self.hash1.wrapping_mul(BASE1).wrapping_add(word.0);
        self.hash2 = self.hash2.wrapping_mul(BASE2).wrapping_add(word.1);
        self.words.push_back(word);
        if self.words.len() > N {
            let old = self.words.pop_front().expect("length checked");
            self.hash1 = self.hash1.wrapping_sub(old.0.wrapping_mul(self.pow1));
            self.hash2 = self.hash2.wrapping_sub(old.1.wrapping_mul(self.pow2));
        }
        (self.words.len() == N).then_some(((self.hash1 as u128) << 64) | self.hash2 as u128)
    }
}

fn hash_word(bytes: &[u8]) -> (u64, u64) {
    let mut h1 = 0xcbf29ce484222325u64;
    let mut h2 = 0x84222325cbf29ce4u64;
    for &raw in bytes {
        let b = raw.to_ascii_lowercase();
        h1 ^= b as u64;
        h1 = h1.wrapping_mul(0x100000001b3);
        h2 ^= (b as u64).wrapping_add(0x9d);
        h2 = h2.wrapping_mul(0x100000001b3);
    }
    (h1, h2)
}

fn feed_segment<F: FnMut(u128)>(segment: &[u8], rolling: &mut RollingGram, mut on_gram: F) {
    let mut start: Option<usize> = None;
    for (i, &b) in segment.iter().enumerate() {
        if b.is_ascii_alphanumeric() {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start.take() {
            if let Some(gram) = rolling.push(hash_word(&segment[s..i])) {
                on_gram(gram);
            }
        }
    }
    if let Some(s) = start {
        if let Some(gram) = rolling.push(hash_word(&segment[s..])) {
            on_gram(gram);
        }
    }
}

fn trim_line(mut line: &[u8]) -> &[u8] {
    while matches!(line.last(), Some(b'\n' | b'\r')) {
        line = &line[..line.len() - 1];
    }
    line
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn feed_training_line<F: FnMut(u128)>(line: &[u8], rolling: &mut RollingGram, mut on_gram: F) {
    let mut rest = line;
    loop {
        if let Some(pos) = find_subslice(rest, EOT) {
            feed_segment(&rest[..pos], rolling, &mut on_gram);
            rolling.clear();
            rest = &rest[pos + EOT.len()..];
        } else {
            feed_segment(rest, rolling, &mut on_gram);
            return;
        }
    }
}

fn load_eval(
    path: &Path,
) -> io::Result<(
    Vec<String>,
    Vec<u64>,
    HashMap<u128, u32>,
    HashMap<u128, Vec<u32>>,
)> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut line = Vec::new();
    let mut ids: Vec<String> = Vec::new();
    let mut gram_counts: Vec<u64> = Vec::new();
    let mut owner: HashMap<u128, u32> = HashMap::new();
    let mut duplicates: HashMap<u128, Vec<u32>> = HashMap::new();
    let mut current: Option<u32> = None;
    let mut rolling = RollingGram::new();
    let mut local: HashSet<u128> = HashSet::new();

    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        let trimmed = trim_line(&line);
        if trimmed.starts_with(START) {
            if current.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "nested eval document marker",
                ));
            }
            let id = String::from_utf8(trimmed[START.len()..].to_vec())
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "eval id is not UTF-8"))?;
            current = Some(ids.len() as u32);
            ids.push(id);
            gram_counts.push(0);
            rolling.clear();
            local.clear();
            continue;
        }
        if trimmed == END {
            let doc = current.take().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "end marker outside eval document",
                )
            })?;
            gram_counts[doc as usize] = local.len() as u64;
            for gram in local.drain() {
                match owner.get(&gram).copied() {
                    None => {
                        owner.insert(gram, doc);
                    }
                    Some(previous) if previous != doc => {
                        let docs = duplicates.entry(gram).or_insert_with(|| vec![previous]);
                        if docs.last().copied() != Some(doc) && !docs.contains(&doc) {
                            docs.push(doc);
                        }
                    }
                    Some(_) => {}
                }
            }
            rolling.clear();
            continue;
        }
        if current.is_none() {
            if !trimmed.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "text outside eval document",
                ));
            }
            continue;
        }
        feed_segment(&line, &mut rolling, |gram| {
            local.insert(gram);
        });
    }
    if current.is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unterminated eval document",
        ));
    }
    Ok((ids, gram_counts, owner, duplicates))
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} EVAL_DOCS.txt REPORT.tsv TRAIN.txt [TRAIN2.txt ...]",
            args[0]
        );
        std::process::exit(2);
    }
    let eval_path = Path::new(&args[1]);
    let report_path = Path::new(&args[2]);
    let train_paths = &args[3..];

    let (ids, eval_gram_counts, owner, duplicates) = load_eval(eval_path)?;
    let mut matched: Vec<HashSet<u128>> = (0..ids.len()).map(|_| HashSet::new()).collect();
    let mut first_source: Vec<Option<String>> = vec![None; ids.len()];
    let mut train_grams = 0u64;
    let mut rolling = RollingGram::new();
    let mut line = Vec::new();

    eprintln!(
        "loaded {} eval docs, {} unique gram owners, {} shared grams",
        ids.len(),
        owner.len(),
        duplicates.len()
    );
    for path_str in train_paths {
        let path = Path::new(path_str);
        let file = File::open(path)?;
        let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
        rolling.clear();
        loop {
            line.clear();
            if reader.read_until(b'\n', &mut line)? == 0 {
                break;
            }
            feed_training_line(&line, &mut rolling, |gram| {
                train_grams += 1;
                if let Some(&doc) = owner.get(&gram) {
                    matched[doc as usize].insert(gram);
                    if first_source[doc as usize].is_none() {
                        first_source[doc as usize] = Some(path_str.clone());
                    }
                    if let Some(extra_docs) = duplicates.get(&gram) {
                        for &extra in extra_docs {
                            matched[extra as usize].insert(gram);
                            if first_source[extra as usize].is_none() {
                                first_source[extra as usize] = Some(path_str.clone());
                            }
                        }
                    }
                }
            });
        }
        eprintln!(
            "scanned {} ({} train grams total)",
            path.display(),
            train_grams
        );
    }

    let file = File::create(report_path)?;
    let mut out = BufWriter::new(file);
    writeln!(out, "# alpha exact 13-gram audit v1")?;
    writeln!(
        out,
        "# normalization=lowercase ASCII alphanumeric words; document boundaries respected"
    )?;
    writeln!(out, "# eval_docs={} train_grams={}", ids.len(), train_grams)?;
    writeln!(
        out,
        "eval_id\tmatched_unique_grams\teval_unique_grams\tfirst_train_source"
    )?;
    let mut contaminated = 0usize;
    for (i, id) in ids.iter().enumerate() {
        if matched[i].is_empty() {
            continue;
        }
        contaminated += 1;
        writeln!(
            out,
            "{}\t{}\t{}\t{}",
            id,
            matched[i].len(),
            eval_gram_counts[i],
            first_source[i].as_deref().unwrap_or("")
        )?;
    }
    out.flush()?;
    eprintln!(
        "contaminated eval docs: {}/{}; report={}",
        contaminated,
        ids.len(),
        report_path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grams(text: &[u8]) -> Vec<u128> {
        let mut rolling = RollingGram::new();
        let mut result = Vec::new();
        feed_segment(text, &mut rolling, |gram| result.push(gram));
        result
    }

    #[test]
    fn same_window_hashes_identically_after_different_prefixes() {
        let window = b"one two three four five six seven eight nine ten eleven twelve thirteen";
        let direct = grams(window);
        let prefixed = grams(
            b"unrelated prefix words then one two three four five six seven eight nine ten eleven twelve thirteen",
        );
        assert_eq!(direct.len(), 1);
        assert!(prefixed.contains(&direct[0]));
    }

    #[test]
    fn end_of_text_prevents_cross_document_grams() {
        let mut rolling = RollingGram::new();
        let mut result = Vec::new();
        feed_training_line(
            b"one two three four five six seven eight nine ten eleven twelve <|end_of_text|> alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu",
            &mut rolling,
            |gram| result.push(gram),
        );
        assert_eq!(result.len(), 1);
    }
}

# Instructions
To clone the project:
``` bash
cd your-project
git clone https://github.com/bentdixon/labl-utils.git
cd labl-utils
# Make new virtual environment (Conda or Venv, etc.), or use pre-existing environment. If using pre-existing, use --no-deps. If creating new, use default.
pip install -e .  # OR, pip install -e . --no-deps
```
To install via `pip`:
``` bash
pip install git+https://github.com/bentdixon/labl-utils.git
```
To utilize the CLI scripts which require the vLLM dependency:
```bash
pip install "labl-utils[cli] @ git+https://github.com/bentdixon/labl-utils.git" 
```

## Core Modules

Example usages of the main modules under `labl/`.

### `transcripts.py`

`Transcript` provides iteration over transcript directories, automatic text parsing, and metadata extraction (site, clinical status, patient ID, session, etc.).

``` python
from labl.transcripts import Transcript

# Set the transcript directory first, containing CHR and HC subdirectories.
# Recommended but not enforced — enables list_transcripts() and robust path handling.
Transcript.set_directory_path("/data/transcripts")

# Iterate through all transcripts
for transcript in Transcript.list_transcripts():
    print(transcript.filename)
    print(len(transcript.participant_lines))

# Define transcript objects directly
t1 = Transcript("CHR/subject_001/interview.txt")
t2 = Transcript("HC/subject_042/interview.txt")

# Access properties
print(t1.patient_id)             # Patient ID string (AMP SCZ filename format)
print(t1.group_status)           # ClinicalGroup.CHR
print(len(t1.lines))             # Total line count
print(len(t1.participant_lines)) # Participant utterances only
print(len(t1.interviewer_lines)) # Interviewer utterances only

# Each line is a TranscriptLine(line_index, speaker, timestamp, text)
for line in t1.participant_lines:
    print(f"{line.line_index}: {line.timestamp} - {line.text}")
```

### `frequency.py`

`FrequencyLookup` loads a frequency corpus (SUBTLEX-US or simple two-column CSV) and provides word-level frequency lookup. The corpus is loaded lazily on first access and cached for the lifetime of the object.

``` python
from labl.frequency import FrequencyLookup

# Initialize with a SUBTLEX-US corpus file
freq = FrequencyLookup("data/frequency_corpus/SUBTLEX-US.csv")

print(f"Loaded {len(freq):,} words from corpus")

# Look up a single word — returns log frequency by default
log_freq = freq.lookup("dog", use_log=True)
raw_freq = freq.lookup("dog", use_log=False)
print(f"dog: log={log_freq:.4f}, raw={raw_freq:.8f}")

# Returns None (or a specified default) for words not in the corpus
score = freq.lookup("xyzzy", default=0.0)

# Check corpus membership
print("hello" in freq)          # True/False
print(freq.contains("hello"))   # equivalent

# Look up multiple words at once
words = ["the", "quick", "brown", "fox"]
scores = freq.lookup_many(words, use_log=True)
for word, score in zip(words, scores):
    print(f"{word:15s} {score}")

# Compute mean frequency for a list of words
# Returns (mean_freq, words_found, words_missing)
text = "The quick brown fox jumps over the lazy dog"
words = text.lower().split()
mean_freq, found, missing = freq.mean_frequency(words)
print(f"Mean log frequency: {mean_freq:.4f}")
print(f"Coverage: {found}/{found + missing} words found in corpus")

# Integrate with Transcript objects
from labl.transcripts import Transcript

Transcript.set_directory_path("/data/transcripts")

for transcript in Transcript.list_transcripts():
    words = [
        word
        for line in transcript.participant_lines
        for word in line.text.split()
    ]
    mean_freq, found, missing = freq.mean_frequency(words)
    if mean_freq is not None:
        coverage = found / (found + missing)
        print(f"{transcript.patient_id}: mean_freq={mean_freq:.4f}, coverage={coverage:.2%}")
```

### `specificity.py`

`SpecificityLookup` computes Specificity 3 (Bolognesi et al., 2020) using WordNet hypernym depth. Higher scores indicate more specific (less generic) concepts. WordNet data is downloaded automatically if not present. Results are cached via `lru_cache` to avoid redundant lookups across a session.

``` python
from labl.specificity import SpecificityLookup

# Raw 0-1 scale (0 = maximally generic, 1 = maximally specific)
spec = SpecificityLookup(pos='n', normalized=False)

# Normalized 0-5 scale (to match Brysbaert concreteness ratings)
spec_norm = SpecificityLookup(pos='n', normalized=True)

# Look up a single noun
score = spec.lookup("dog")
print(f"dog (raw): {score:.4f}")

score_norm = spec_norm.lookup("dog")
print(f"dog (normalized): {score_norm:.4f}")

# Returns None (or a specified default) for words not found in WordNet
score = spec.lookup("xyzzy", default=0.0)

# Check WordNet membership
print("animal" in spec)          # True/False
print(spec.contains("animal"))   # equivalent

# Look up multiple words at once
words = ["entity", "object", "animal", "dog", "terrier"]
scores = spec.lookup_many(words)
for word, score in zip(words, scores):
    label = f"{score:.4f}" if score is not None else "NOT FOUND"
    print(f"{word:15s} {label}")

# Override part-of-speech per lookup (instance pos defaults to 'n')
verb_score = spec.lookup("run", pos='v')

# Inspect and manage the internal cache
print(spec.cache_info())
spec.clear_cache()

# Integrate with Transcript objects using spaCy for POS tagging
import spacy
from labl.transcripts import Transcript, ClinicalGroup

nlp = spacy.load("en_core_web_sm")
Transcript.set_directory_path("/data/transcripts")

for transcript in Transcript.list_transcripts():
    texts = [line.text for line in transcript.participant_lines]
    scores = []

    for doc in nlp.pipe(texts, batch_size=200):
        for token in doc:
            if token.pos_ == "NOUN" and token.is_alpha and not token.is_stop:
                s = spec.lookup(token.text.lower())
                if s is not None:
                    scores.append(s)

    if scores:
        mean_spec = sum(scores) / len(scores)
        print(f"{transcript.patient_id} ({transcript.group_status.value}): mean_specificity={mean_spec:.4f}")
```

## Using `features/`

### `combine_data.py`

Used to efficiently combine CSVs with `patient_id` columns, and optionally `clinical_status`, appending all other columns afterwards. Uses Polars to support memory-efficient loading and speedy operations.

``` python
from labl.features.combine_data import combine_csvs

# Expects CSV with `patient_id` column, optionally `clinical_status`, and then metric columns
c1 = "some/path/data.csv"  # `patient_id`, `metric_1`
c2 = "some/path/data.csv"  # `patient_id`, `clinical_status`, `metric_2`

# Results in combined Polars DataFrame with `patient_id`, `clinical_status`, `metric_1`, `metric_2`
# matched on `patient_id`
combined_data = combine_csvs(c1, c2)
```

## Using `cli/`

CLI scripts are available as console commands after installing the package (`pip install -e .`). The `vllm` dependency required by the LLM scripts is optional and can be installed with `pip install -e ".[cli]"`.

### `labl-assign-roles`

Classifies PARTICIPANT and INTERVIEWER speaker tags using vLLM batch inference.

``` bash
labl-assign-roles --i "input/transcripts/" --o "output/" --thinking "medium" --gpu 0 --tp 1 --batch-size 32
```

### `labl-assign-interview-type`

Classifies interview type as OPEN or PSYCHS using vLLM batch inference.

``` bash
labl-assign-interview-type --i "input/transcripts/" --o "output/" --thinking "medium" --gpu 0 --tp 1 --batch-size 32
```

### `labl-structure`

Organizes transcripts by language and clinical status. Optionally uses CSV for clinical status mapping.

``` bash
labl-structure --i "input/transcripts/" --o "output/" --text-type "open" --gpu 0
labl-structure --i "input/" --o "output/" --text-type "psychs" --csv "status.csv" --gpu 0
```

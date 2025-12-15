'''
Skeleton script for looking up word frequencies given corpuses in each language. 
Assumes a word, word_freq column format. 
'''


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import csv
from functools import cache
from typing import Optional

from data.langs import Language
from utils.transcripts import Transcript


DATASETS: dict[tuple[Language, str], str] = {
    (Language.en, "written"): "path/to/english_written_freqs.csv",
    (Language.en, "spoken"): "path/to/english_spoken_freqs.csv",
    # ...
}


@cache
def load_frequency_dict(filepath: str) -> dict[str, float]:
    """Load a word frequency CSV into a dictionary for fast lookup."""
    freq_dict: dict[str, float] = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                word = row[0].lower()
                try:
                    freq_dict[word] = float(row[1])
                except ValueError:
                    continue
    return freq_dict


def lookup(lang: Language, word: str) -> tuple[Optional[float], Optional[float]]:
    """
    Look up word frequency in written and spoken corpora.
    
    Returns:
        Tuple of (written_freq, spoken_freq), with None for missing values.
    """
    word_lower = word.lower()

    written_freq: Optional[float] = None
    spoken_freq: Optional[float] = None

    if (lang, "written") in DATASETS:
        written_dict = load_frequency_dict(DATASETS[(lang, "written")])
        written_freq = written_dict.get(word_lower)

    if (lang, "spoken") in DATASETS:
        spoken_dict = load_frequency_dict(DATASETS[(lang, "spoken")])
        spoken_freq = spoken_dict.get(word_lower)

    return (written_freq, spoken_freq)


def main() -> None:
    Transcript.set_directory_path("some/path/directory")
    transcripts = Transcript.list_transcripts()

    for t in transcripts:
        for line in t.participant_lines:
            for word in line.text.split():
                written_freq, spoken_freq = lookup(t.language, word)
                # process frequencies as needed


if __name__ == "__main__":
    main()
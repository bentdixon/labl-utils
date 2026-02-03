"""
Class-based utility for looking up word frequencies from corpus files.
Supports SUBTLEX-US and other frequency corpora.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import csv
import re
from pathlib import Path
from functools import cached_property
from typing import Optional

from data.langs import Language


class FrequencyLookup:
    def __init__(
        self,
        filepath: str | Path,
        corpus_type: str = "subtlex",
        language: Language = Language.en
    ):
        self.filepath = Path(filepath)
        self.corpus_type = corpus_type.lower()
        self.language = language

        if not self.filepath.exists():
            raise FileNotFoundError(f"Frequency file not found: {self.filepath}")

    @cached_property
    def _frequency_data(self) -> dict[str, tuple[float, float]]:
        if self.corpus_type == "subtlex":
            return self._load_subtlex()
        else:
            return self._load_simple()

    def _load_subtlex(self) -> dict[str, tuple[float, float]]:
        """
        Load SUBTLEX-style corpus with FREQcount and Lg10WF columns.
        """
        freq_dict: dict[str, tuple[float, float]] = {}

        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            rows = list(reader)
            total_count = sum(float(row.get('FREQcount', 0)) for row in rows)

            for row in rows:
                word = row.get('Word', '').lower().strip()
                if not word:
                    continue
                try:
                    freq_count = float(row.get('FREQcount', 0))
                    log_freq = float(row.get('Lg10WF', 0))
                    raw_freq = freq_count / total_count if total_count > 0 else 0
                    freq_dict[word] = (raw_freq, log_freq)
                except (ValueError, KeyError):
                    continue

        return freq_dict

    def _load_simple(self) -> dict[str, tuple[float, float]]:
        freq_dict: dict[str, tuple[float, float]] = {}

        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header

            for row in reader:
                if len(row) >= 2:
                    word = row[0].lower().strip()
                    try:
                        raw_freq = float(row[1])
                        log_freq = float(row[2]) if len(row) > 2 else 0
                        freq_dict[word] = (raw_freq, log_freq)
                    except ValueError:
                        continue

        return freq_dict

    def lookup(
        self,
        word: str,
        use_log: bool = True,
        default: Optional[float] = None
    ) -> Optional[float]:
        word_lower = word.lower().strip()

        if word_lower in self._frequency_data:
            raw_freq, log_freq = self._frequency_data[word_lower]
            return log_freq if use_log else raw_freq

        return default

    def lookup_many(
        self,
        words: list[str],
        use_log: bool = True,
        default: Optional[float] = None
    ) -> list[Optional[float]]:
        return [self.lookup(word, use_log=use_log, default=default) for word in words]

    def mean_frequency(
        self,
        words: list[str],
        use_log: bool = True,
        remove_punctuation: bool = True
    ) -> tuple[Optional[float], int, int]:
        if remove_punctuation:
            words = [re.sub(r'[^\w\s]', '', word.lower()) for word in words]

        frequencies = []
        words_missing = 0

        for word in words:
            freq = self.lookup(word, use_log=use_log)
            if freq is not None:
                frequencies.append(freq)
            else:
                words_missing += 1

        words_found = len(frequencies)

        if words_found == 0:
            return None, 0, words_missing

        mean_freq = sum(frequencies) / words_found
        return mean_freq, words_found, words_missing

    def contains(self, word: str) -> bool:
        """
        Check if a word exists in the frequency corpus.
        """
        return word.lower().strip() in self._frequency_data

    @property
    def size(self) -> int:
        """Return the number of unique words in the corpus."""
        return len(self._frequency_data)

    def __len__(self) -> int:
        """Return the number of unique words in the corpus."""
        return self.size

    def __contains__(self, word: str) -> bool:
        """Support 'word in freq_lookup' syntax."""
        return self.contains(word)


def main() -> None:
    """Example usage of FrequencyLookup class."""
    from utils.transcripts import Transcript

    # Initialize frequency lookup with SUBTLEX-US
    freq = FrequencyLookup("data/frequency_corpus/SUBTLEX-US.csv")

    print(f"Loaded {len(freq):,} words from corpus")
    print()

    # Example 1: Look up individual words
    test_words = ["hello", "world", "python", "extraordinarily"]
    for word in test_words:
        log_freq = freq.lookup(word, use_log=True)
        raw_freq = freq.lookup(word, use_log=False)
        if log_freq is not None:
            print(f"{word:20s} log: {log_freq:8.4f}  raw: {raw_freq:.8f}")
        else:
            print(f"{word:20s} NOT FOUND")
    print()

    # Example 2: Calculate mean frequency for a text
    text = "The quick brown fox jumps over the lazy dog"
    words = text.lower().split()
    mean_freq, found, missing = freq.mean_frequency(words)
    print(f"Text: '{text}'")
    print(f"Mean log frequency: {mean_freq:.4f}")
    print(f"Words found: {found}/{found+missing}")
    print()

    # Example 3: Process transcripts
    # Uncomment to use with transcripts:
    # Transcript.set_directory_path("some/path/directory")
    # for transcript in Transcript.list_transcripts():
    #     words = [word for line in transcript.participant_lines
    #              for word in line.text.split()]
    #     mean_freq, found, missing = freq.mean_frequency(words)
    #     print(f"{transcript.patient_id}: mean_freq={mean_freq:.4f}, coverage={found/(found+missing):.2%}")


if __name__ == "__main__":
    main()
"""
Class-based utility for computing word specificity using WordNet.
Based on Bolognesi et al. (2020) Specificity 3 measure.

Specificity 3 measures how specific (vs. generic) a concept is based on its
position in the WordNet hypernym hierarchy. Higher values = more specific.

Formula: Specificity_3 = (1 + d) / max_depth
Where:
    d = number of direct and indirect hypernyms (ancestors up to root)
    max_depth = 20 (maximum depth of WordNet 3.0 noun taxonomy)

Raw output: 0-1 scale (0 = generic like "entity", 1 = maximally specific)
Normalized output: 0-5 scale (to match Brysbaert concreteness ratings)

Reference:
    Bolognesi, M., Burgers, C., & Caselli, T. (2020).
    "On abstraction: decoupling conceptual concreteness and categorical specificity."
    Cognitive Processing, 21, 365-381.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from functools import lru_cache
from typing import Optional
from nltk.corpus import wordnet as wn


WN_MAX_DEPTH = 20


class SpecificityLookup:
    def __init__(self, pos: str = 'n', normalized: bool = False):
        """
        Initialize specificity lookup.

        Args:
            pos: Part-of-speech for WordNet lookup ('n', 'v', 'a', 'r')
            normalized: If True, return scores on 0-5 scale; if False, 0-1 scale
        """
        self.pos = pos
        self.normalized = normalized

        try:
            wn.synsets('test')
        except LookupError:
            import nltk
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)

    @staticmethod
    @lru_cache(maxsize=10000)
    def _get_hypernym_depth(word: str, pos: str) -> Optional[int]:
        """
        Get the number of hypernyms (ancestors) for a word's first sense.
        """
        synsets = wn.synsets(word, pos=pos)
        if not synsets:
            return None

        first_sense = synsets[0]
        hypernym_closure = lambda s: s.hypernyms()
        all_hypernyms = list(first_sense.closure(hypernym_closure))

        if len(all_hypernyms) == 0:
            instance_hypernyms = first_sense.instance_hypernyms()
            if instance_hypernyms:
                all_hypernyms = list(instance_hypernyms[0].closure(hypernym_closure))
                all_hypernyms = instance_hypernyms + all_hypernyms

        return len(all_hypernyms)

    def lookup(
        self,
        word: str,
        pos: Optional[str] = None,
        default: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate specificity score for a word.

        Args:
            word: The word to look up
            pos: Part-of-speech override (uses instance default if None)
            default: Default value if word not found in WordNet

        Returns:
            Specificity score or default if not found
        """
        word_lower = word.lower().strip()
        pos_to_use = pos if pos is not None else self.pos

        depth = self._get_hypernym_depth(word_lower, pos_to_use)
        if depth is None:
            return default

        raw_score = (depth + 1) / WN_MAX_DEPTH

        if self.normalized:
            return raw_score * 5

        return raw_score

    def lookup_many(
        self,
        words: list[str],
        pos: Optional[str] = None,
        default: Optional[float] = None
    ) -> list[Optional[float]]:
        """
        Calculate specificity scores for multiple words.
        """
        return [self.lookup(word, pos=pos, default=default) for word in words]

    def contains(self, word: str, pos: Optional[str] = None) -> bool:
        """
        Check if a word exists in WordNet for the given part of speech.
        """
        word_lower = word.lower().strip()
        pos_to_use = pos if pos is not None else self.pos
        synsets = wn.synsets(word_lower, pos=pos_to_use)
        return len(synsets) > 0

    def clear_cache(self) -> None:
        """
        Clear the internal LRU cache for hypernym depth lookups.
        """
        self._get_hypernym_depth.cache_clear()

    def cache_info(self):
        """
        Get cache statistics for the hypernym depth lookups.
        """
        return self._get_hypernym_depth.cache_info()

    def __contains__(self, word: str) -> bool:
        """Support 'word in spec_lookup' syntax."""
        return self.contains(word)


def main() -> None:
    """Example usage of SpecificityLookup class."""

    spec = SpecificityLookup(pos='n', normalized=False)

    print("Specificity scores for nouns (raw 0-1 scale):")
    test_words = ["entity", "object", "animal", "dog", "terrier", "cat", "python"]
    for word in test_words:
        score = spec.lookup(word)
        if score is not None:
            print(f"{word:20s} {score:.4f}")
        else:
            print(f"{word:20s} NOT FOUND")
    print()

    spec_norm = SpecificityLookup(pos='n', normalized=True)
    print("Specificity scores for nouns (normalized 0-5 scale):")
    for word in test_words:
        score = spec_norm.lookup(word)
        if score is not None:
            print(f"{word:20s} {score:.4f}")
        else:
            print(f"{word:20s} NOT FOUND")
    print()

    print(f"Cache info: {spec.cache_info()}")


if __name__ == "__main__":
    main()
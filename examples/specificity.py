"""
Specificity 3 Implementation
Based on: Bolognesi, M., Burgers, C., & Caselli, T. (2020). 
"On abstraction: decoupling conceptual concreteness and categorical specificity."
Cognitive Processing, 21, 365-381.

Specificity 3 measures how specific (vs. generic) a concept is based on its
position in the WordNet hypernym hierarchy. Higher values = more specific.

Formula: Specificity_3 = (1 + d) / max_depth
Where:
    d = number of direct and indirect hypernyms (ancestors up to root)
    max_depth = 20 (maximum depth of WordNet 3.0 noun taxonomy)

Raw output: 0-1 scale (0 = generic like "entity", 1 = maximally specific)
Normalized output: 0-5 scale (to match Brysbaert concreteness ratings)
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ''  # Set yourself

import spacy
import numpy as np
import polars as pl
from nltk.corpus import wordnet as wn
from typing import Optional, Tuple
from utils.utils.transcript_utils import Transcript, ClinicalGroup
from functools import lru_cache


WN_MAX_DEPTH = 20


def get_hypernym_depth(word: str, pos: str = 'n') -> Optional[int]:
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


@lru_cache(maxsize=None)
def specificity_3(word: str, normalized: bool = False) -> Optional[float]:
    """
    Calculate Specificity 3 for a word.
    """
    depth = get_hypernym_depth(word)
    if depth is None:
        return None
    
    raw_score = (depth + 1) / WN_MAX_DEPTH
    
    if normalized:
        return raw_score * 5
    
    return raw_score


def process_transcript(transcript: Transcript, nlp) -> Tuple[str, float]:
    """Process a single transcript and return (patient_id, average_specificity)."""
    specificity_scores = []
    
    texts = [line.text for line in transcript.participant_lines]
    
    for doc in nlp.pipe(texts, batch_size=200):
        for token in doc:
            if token.pos_ == 'NOUN' and token.is_alpha and not token.is_stop:
                score = specificity_3(token.text.lower())
                if score is not None:
                    specificity_scores.append(score)
    
    if specificity_scores:
        average_specificity = float(np.mean(specificity_scores))
    else:
        average_specificity = float('nan')
    
    return (transcript.patient_id, average_specificity)


if __name__ == "__main__":
    nlp = spacy.load('en_core_web_sm')
    Transcript.set_directory_path("/data/transcripts/")

    if Transcript.directory_path is None:
        raise ValueError("Transcript directory not set")

    chr_transcripts = [t for t in Transcript.list_transcripts() if t.group_status == ClinicalGroup.CHR]
    hc_transcripts = [t for t in Transcript.list_transcripts() if t.group_status == ClinicalGroup.HC]

    for group, status in [(chr_transcripts, 'CHR'), (hc_transcripts, 'HC')]:
        print(f"\nGroup: {status} ({len(group)} transcripts)")
        results = []

        for transcript in group:
            result = process_transcript(transcript, nlp)
            results.append(result)
            print(f"\tDone: {result[0]}")

        df = pl.DataFrame(results, orient="row", schema=["patient_id", "specificity"])
        print(df)
        dest = f"/some/output/specificity_{status}.csv"
        df.write_csv(dest)
        print(f"Saved to {dest}")

    print(f"\n{specificity_3.cache_info()}")  # Confirm cache performance improvement
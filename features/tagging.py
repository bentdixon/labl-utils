"""
For every participant, tallies total feature count derived from Stanza
and creates an output CSV where:

            clinical_status feat1 feat2 ...
patient1    CHR             119   16
patient2    CHR             38    9
patient3    HC              13    NaN
...

Refactored to use Transcript class abstraction.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import math
import argparse
import numpy as np
import stanza
from pathlib import Path
from typing import Optional

from utils.transcripts import Transcript, ClinicalGroup
from data.langs import Language


# Languages with Stanza support
SUPPORTED_STANZA_LANGUAGES = {'zh', 'es', 'en', 'ko', 'it', 'ja', 'da', 'de', 'fr', 'yue'}


def extract_feature(featstr: Optional[str], feat_type: str) -> str:
    """Extract specific morphological feature from Stanza output."""
    feature = ''
    if featstr is not None:
        feature_list = featstr.split('|')
        for f in feature_list:
            if feat_type in f:
                if feat_type != 'Mood':
                    feature = f[len(feat_type)+1:]
                else:
                    feature = f[len(feat_type)+1:] + '_mood'
    return feature


def readin_word_freqs(inputfilename: str) -> dict[str, float]:
    """Read word frequencies from file and return log-transformed values."""
    wordfreqs = {}
    with open(inputfilename, 'r', encoding='latin-1') as infile:
        for row in infile:
            row = row.rstrip('\n')
            parts = row.split('\t')
            if len(parts) >= 2:
                wordfreqs[parts[0]] = math.log(int(parts[1]))
    return wordfreqs


def build_tag_feat_dict(tags_inputfile: str) -> dict[str, int]:
    """Build dictionary of feature tags initialized to zero."""
    tag_feat_dict = {}
    with open(tags_inputfile, 'r') as tags_infile:
        for tag in tags_infile:
            tag = tag.rstrip('\n')
            tag_feat_dict[tag] = 0
    return tag_feat_dict


def determine_freqs(word, wordfreqs: dict[str, float], word_freq_list: list) -> list:
    """Look up word frequency for content words."""
    if word.upos in ['NOUN', 'VERB', 'ADV', 'ADJ']:
        pos_word = word.upos + '_' + word.text
        if pos_word in wordfreqs:
            wfreq = wordfreqs[pos_word]
            if wfreq > 0:
                word_freq_list.append([wfreq])
    return word_freq_list


def fill_tag_feat_slots(
    tag_feat_dict: dict[str, int],
    tags: list[list[str]],
    freq_statistics: dict
) -> dict:
    """Count occurrences of each linguistic feature."""
    tally_dict = {key: 0 for key in tag_feat_dict}
    
    # Count tags (skip lemma at index 0)
    if tags:
        for i in range(1, len(tags[0])):
            for tlist in tags:
                if tlist[i] != '' and tlist[i] in tally_dict:
                    tally_dict[tlist[i]] += 1
    
    # Add aggregate statistics
    for stat_name, stat_value in freq_statistics.items():
        tally_dict[stat_name] = stat_value
    
    return tally_dict


def save_tags(
    tally_tags_feat_dict: dict,
    speaker_role: str,
    output_file: Path
) -> None:
    """Save feature counts to TSV file."""
    if not tally_tags_feat_dict:
        print("No data to save.")
        return
    
    keys = list(tally_tags_feat_dict.keys())
    first_key = keys[0]
    
    # Get feature labels, renaming problematic ones
    labels_original = list(tally_tags_feat_dict[first_key].keys())
    labels_renamed = []
    for label in labels_original:
        if label == '1':
            label = 'p1'
        elif label == '2':
            label = 'p2'
        elif label == '3':
            label = 'p3'
        elif label == 'Yes':
            label = 'pronoun_possession'
        labels_renamed.append(label)
    
    header = [
        'patient_id', 'language', 'clinical_group', 'transcript_type',
        'day', 'session', 'speaker_role'
    ] + labels_renamed
    
    with open(output_file, 'w') as outfile:
        outfile.write('\t'.join(header) + '\n')
        
        for key, features in tally_tags_feat_dict.items():
            # Parse key: patient_id_transcript_type_day_session
            parts = key.split('_')
            patient_id = parts[0]
            language = parts[1]
            clinical_group = parts[2]
            transcript_type = parts[3]
            day = parts[4]
            session = parts[5]
            
            row_values = [str(features[label]) for label in labels_original]
            row = [
                patient_id, language, clinical_group, transcript_type,
                day, session, speaker_role
            ] + row_values
            
            outfile.write('\t'.join(row) + '\n')
    
    print(f"Saved output to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tally morphosyntactic features from transcripts using Stanza.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--i", type=str, required=True,
                        help="Input directory containing transcript files")
    parser.add_argument("--o", type=str, required=True,
                        help="Output TSV file path")
    parser.add_argument("--feats", type=str, required=True,
                        help="Path to feature list file (tags_upos_xpos.txt)")
    parser.add_argument("--wordfreqs", type=str, required=False, default=None,
                        help="Path to word frequency file (optional)")
    parser.add_argument("--language", type=str, required=True,
                        choices=[lang.name for lang in Language if lang.name in SUPPORTED_STANZA_LANGUAGES],
                        help="Language code for Stanza pipeline (filters transcripts to this language)")
    parser.add_argument("--speaker", type=str, default="participant",
                        choices=['participant', 'interviewer'],
                        help="Speaker role to analyze")
    parser.add_argument("--gpu", type=int, required=True,
                        help="GPU device ID to use")
    parser.add_argument("--batch_size", type=int, default=400,
                        help="Batch size for Stanza dependency parsing")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    input_dir = Path(args.i)
    output_file = Path(args.o)
    feature_list_path = Path(args.feats)
    target_language = Language[args.language]  # Convert string to Language enum

    # Load word frequencies if provided
    wordfreqs = {}
    if args.wordfreqs:
        wordfreqs = readin_word_freqs(args.wordfreqs)
        print(f"Loaded {len(wordfreqs)} word frequencies")

    # Build feature dictionary
    tag_feat_dict = build_tag_feat_dict(feature_list_path)
    print(f"Loaded {len(tag_feat_dict)} feature tags")

    # Initialize Stanza pipeline
    nlp = stanza.Pipeline(args.language, depparse_batch_size=args.batch_size)
    print(f"Initialized Stanza pipeline for language: {args.language}")

    # Set transcript directory and collect transcripts
    Transcript.set_directory_path(input_dir)
    all_transcripts = Transcript.list_transcripts()
    
    # Filter to target language
    transcripts = [t for t in all_transcripts if t.language == target_language]
    print(f"Found {len(transcripts)} transcripts matching language '{args.language}' (out of {len(all_transcripts)} total)")

    tally_tags_feat_dict = {}

    for i, transcript in enumerate(transcripts):
        print(f"[{i+1}/{len(transcripts)}] Processing: {transcript.filename}")

        # Select speaker lines based on argument
        if args.speaker == "participant":
            lines = transcript.participant_lines
        else:
            print("Getting interviewer lines...")
            lines = transcript.interviewer_lines

        if not lines:
            print(f"  No {args.speaker} lines found, skipping.")
            continue

        tags = []
        num_words = 0
        num_sentences = 0
        word_freq_list = []

        # Process each line
        for transcript_line in lines:
            sentence_text = transcript_line.text
            if not sentence_text.strip():
                continue

            doc = nlp(sentence_text)
            for sent in doc.sentences:  # type: ignore
                num_sentences += 1
                for word in sent.words:
                    case = extract_feature(word.feats, 'Case')
                    number = extract_feature(word.feats, 'Number')
                    person = extract_feature(word.feats, 'Person')
                    gender = extract_feature(word.feats, 'Gender')
                    prontype = extract_feature(word.feats, 'PronType')
                    definite = extract_feature(word.feats, 'Definite')
                    mood = extract_feature(word.feats, 'Mood')
                    tense = extract_feature(word.feats, 'Tense')
                    verbform = extract_feature(word.feats, 'VerbForm')
                    poss = extract_feature(word.feats, 'Poss')
                    ntype = extract_feature(word.feats, 'NumType')

                    tags.append([
                        word.lemma, word.upos, word.xpos, word.deprel,
                        case, number, person, gender, prontype, definite,
                        mood, tense, verbform, poss, ntype
                    ])

                    word_freq_list = determine_freqs(word, wordfreqs, word_freq_list)
                    num_words += 1

        # Calculate mean word frequency
        if word_freq_list:
            mean_word_freq = np.array(word_freq_list).mean()
        else:
            mean_word_freq = np.nan

        # Build unique key for this transcript
        key = '_'.join([
            transcript.patient_id or 'UNKNOWN',
            transcript.language.name if transcript.language else 'UNKNOWN',
            transcript.group_status.value if transcript.group_status else 'UNKNOWN',
            transcript.transcript_type or 'UNKNOWN',
            transcript.day or 'UNKNOWN',
            transcript.session or 'UNKNOWN'
        ])

        freq_statistics = {
            'num_sent': num_sentences,
            'num_words': num_words,
            'word_freq': mean_word_freq,
            'file_name': str(transcript.filename)
        }

        tally_tags_feat_dict[key] = fill_tag_feat_slots(
            tag_feat_dict, tags, freq_statistics
        )

        print(f"  Processed {num_sentences} sentences, {num_words} words")

    # Save results
    save_tags(tally_tags_feat_dict, args.speaker, output_file)


if __name__ == "__main__":
    main()
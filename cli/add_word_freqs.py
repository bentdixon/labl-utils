"""
CLI for tagging features CSV with word frequencies.
"""

import argparse
import polars as pl
from pathlib import Path

from utils.frequency import load_frequency_dict, get_transcript_word_frequency
from utils.transcripts import Transcript


def fill_word_frequencies(
    features_df: pl.DataFrame,
    freq_dict: dict[str, float],
    transcript_dir: Path,
    filename_col: str = "filename",
    output_col: str = "word_freq",
) -> pl.DataFrame:
    """
    Fill word frequency column in features DataFrame.
    
    Args:
        features_df: DataFrame with filename column and empty word_freq column
        freq_dict: Dictionary mapping word -> log frequency
        transcript_dir: Base directory containing transcript files
        filename_col: Name of column containing transcript filenames
        output_col: Name of column to fill with word frequencies
        
    Returns:
        DataFrame with word_freq column filled in
    """
    Transcript.set_directory_path(transcript_dir)
    
    frequencies = []
    total_rows = len(features_df)
    
    for i, filename in enumerate(features_df[filename_col].to_list()):
        filepath = transcript_dir / filename
        
        if not filepath.exists():
            print(f"[{i+1}/{total_rows}] File not found: {filename}")
            frequencies.append(None)
            continue
        
        mean_freq, found, missing = get_transcript_word_frequency(filepath, freq_dict)
        
        if mean_freq is None:
            print(f"[{i+1}/{total_rows}] No words found in dictionary: {filename}")
        else:
            coverage = found / (found + missing) * 100 if (found + missing) > 0 else 0
            print(f"[{i+1}/{total_rows}] {filename}: mean_log_freq={mean_freq:.4f} ({found} words, {coverage:.1f}% coverage)")
        
        frequencies.append(mean_freq)
    
    # Replace or add the output column
    if output_col in features_df.columns:
        features_df = features_df.drop(output_col)
    
    features_df = features_df.with_columns(
        pl.Series(name=output_col, values=frequencies)
    )
    
    return features_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill word frequency column in features CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input features CSV with empty word_freq column",
    )
    parser.add_argument(
        "--freq",
        type=str,
        required=True,
        help="Word frequency CSV with 'word' and 'log_frequency' columns",
    )
    parser.add_argument(
        "--transcripts",
        type=str,
        required=True,
        help="Base directory containing transcript files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Output CSV path (defaults to overwriting input)",
    )
    parser.add_argument(
        "--filename-col",
        type=str,
        default="filename",
        help="Name of column containing transcript filenames",
    )
    parser.add_argument(
        "--output-col",
        type=str,
        default="word_freq",
        help="Name of column to fill with word frequencies",
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    freq_path = Path(args.freq)
    transcript_dir = Path(args.transcripts)
    output_path = Path(args.output) if args.output else input_path
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    if not freq_path.exists():
        print(f"Error: Frequency file not found: {freq_path}")
        return
    
    if not transcript_dir.exists():
        print(f"Error: Transcript directory not found: {transcript_dir}")
        return
    
    print(f"Loading frequency dictionary from {freq_path}...")
    freq_dict = load_frequency_dict(freq_path, use_log=True)
    print(f"Loaded {len(freq_dict):,} words")
    
    print(f"Loading features from {input_path}...")
    features_df = pl.read_csv(input_path)
    print(f"Loaded {len(features_df):,} rows")
    
    print(f"Processing transcripts from {transcript_dir}...")
    result_df = fill_word_frequencies(
        features_df=features_df,
        freq_dict=freq_dict,
        transcript_dir=transcript_dir,
        filename_col=args.filename_col,
        output_col=args.output_col,
    )
    
    result_df.write_csv(output_path)
    print(f"Saved results to {output_path}")
    
    # Summary statistics
    filled = result_df[args.output_col].drop_nulls().len()
    total = len(result_df)
    print(f"\nSummary: {filled}/{total} rows filled ({filled/total*100:.1f}%)")


if __name__ == "__main__":
    main()
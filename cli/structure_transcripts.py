"""
CLI tool - takes a directory path of unlabeled transcripts (by language and clinical status) as input
and creates a properly structured directory in a target location passed as an argument. 

CSV input allows for external CSV with 'patient_id' and 'clinical_status' columns to be used 
for external determination of group status if not availabile from pathname.

Pretty-prints output location and structure first to user, asking for confirmation.

Output structure:

{name}/
    English/
        CHR/
        HC/
    Spanish/
        CHR/
        HC/
        ...
    ...

"""

from pathlib import Path
import argparse
import pprint
from utils.transcripts import ClinicalGroup, Transcript
from utils.determine_language import determine_language
from data.langs import Language


def set_clinical_status(transcript: Transcript) -> None:
    for path in transcript.full_path.iterdir():
        if "CHR" in str(path):
            transcript.group_status = ClinicalGroup.CHR
        elif "HC" in str(path):
            transcript.group_status = ClinicalGroup.HC
        else:
            transcript.group_status = ClinicalGroup.UNKNOWN


def set_language(transcript: Transcript) -> None:
    if transcript.language is not Language.UNKNOWN:  # the first language check appears when each transcript is instantiated, checking site codes
        return
    else:
        transcript.language = determine_language(transcript)
    return


def set_type(transcript: Transcript) -> None:


    pass


def structure_transcripts(dirpath: Path, outpath: Path, csv: Path | None) -> dict:
    txt_files = list(dirpath.rglob('*.txt'))  # finds all text files in the data folder
    for t in txt_files:
        transcript = Transcript(t)
        set_clinical_status(transcript)
        set_language(transcript)
        

    output_struct: dict = {}

    

    return output_struct



def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--i", type=str, required=True)
    parser.add_argument("--o", type=str, required=True)
    parser.add_argument("--csv", type=str, required=False)
    args = parser.parse_args()

    input = Path(args.i)
    output = Path(args.o)

    csv_path = None
    if args.csv is not None:
        csv_path = Path(args.csv)

    structure_transcripts(input, output, csv_path)


if __name__ == "__main__":
    main()
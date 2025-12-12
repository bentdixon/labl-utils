"""
CLI tool - takes a directory path of unlabeled transcripts (by language and clinical status) as input
and creates a properly structured directory in a target location passed as an argument.

CSV input allows for external CSV with 'patient_id' and 'clinical_status' columns to be used
for external determination of group status if not availabile from pathname.

Pretty-prints output location and structure first to user, asking for confirmation.

Output structure:

{name}/
    {text_type: open, psychs, diaries}/
        English/
            CHR/
            HC/
        Spanish/
            CHR/
            HC/
        ...
    ...

"""
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'

# Set GPU before any CUDA imports
if '--gpu' in sys.argv:
    gpu_idx = sys.argv.index('--gpu')
    if gpu_idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[gpu_idx + 1]

import csv
from pathlib import Path
import argparse
import shutil
from rich.console import Console
from rich.tree import Tree
from rich.prompt import Confirm
from utils.transcripts import ClinicalGroup, Transcript
from utils.determine_language import determine_language
from data.langs import Language, SITE_CODE_TO_LANGUAGES

console = Console()


def load_clinical_status_csv(csv_path: Path) -> dict[str, ClinicalGroup]:
    """Load CSV and return mapping of patient_id to ClinicalGroup."""
    status_map: dict[str, ClinicalGroup] = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        if 'patient_id' not in reader.fieldnames or 'clinical_status' not in reader.fieldnames:
            raise ValueError("CSV must contain 'patient_id' and 'clinical_status' columns")

        for row in reader:
            patient_id = row['patient_id'].strip()
            status_str = row['clinical_status'].strip().upper()

            if status_str == 'CHR':
                status_map[patient_id] = ClinicalGroup.CHR
            elif status_str == 'HC':
                status_map[patient_id] = ClinicalGroup.HC
            else:
                status_map[patient_id] = ClinicalGroup.UNKNOWN

    return status_map


def set_clinical_status(transcript: Transcript, status_map: dict[str, ClinicalGroup] | None = None) -> None:
    """
    Set clinical status from CSV if available, otherwise infer from path.
    """
    if status_map is not None and transcript.patient_id in status_map:
        transcript.group_status = status_map[transcript.patient_id]
        return

    if status_map is not None:
        console.print(f"[yellow]Warning:[/yellow] Patient {transcript.patient_id} not found in CSV, inferring from path.")

    for path in transcript.full_path.parents:
        if "CHR" in path.name.upper():
            transcript.group_status = ClinicalGroup.CHR
            return
        elif "HC" in path.name.upper():
            transcript.group_status = ClinicalGroup.HC
            return

    transcript.group_status = ClinicalGroup.UNKNOWN


def set_language(transcript: Transcript) -> None:
    language = determine_language(transcript)

    if transcript.site is None:
        console.print(f"[yellow]Warning:[/yellow] Transcript {transcript.filename} has no site code.")
        transcript.language = language
    else:
        langs = SITE_CODE_TO_LANGUAGES.get(transcript.site, (Language.UNKNOWN,))
        if language in langs: 
            transcript.language = language
        else:
            print("Language not found.")
            print(f"language: {language}")
            print(f"transcript: {transcript.filename}")
            print(f"transcript: {transcript.site}")
            print(f"languages in site: {langs}")
            exit(1)
            transcript.language = Language.UNKNOWN


def structure_transcripts(
    dirpath: Path,
    csv_path: Path | None,
    text_type: str
) -> dict[str, dict[str, dict[str, list[Transcript]]]]:
    """
    Build nested structure: {text_type: {language_name: {clinical_group: [transcripts]}}}
    """
    txt_files = list(dirpath.rglob('*.txt'))

    status_map: dict[str, ClinicalGroup] | None = None
    if csv_path is not None:
        status_map = load_clinical_status_csv(csv_path)

    output_struct: dict[str, dict[str, dict[str, list[Transcript]]]] = {}

    for t in txt_files:
        transcript = Transcript(t)
        set_clinical_status(transcript, status_map)
        set_language(transcript)
        
        lang_name = str(transcript.language)
        group_name = str(transcript.group_status)

        if text_type not in output_struct:
            output_struct[text_type] = {}
        if lang_name not in output_struct[text_type]:
            output_struct[text_type][lang_name] = {}
        if group_name not in output_struct[text_type][lang_name]:
            output_struct[text_type][lang_name][group_name] = []

        output_struct[text_type][lang_name][group_name].append(transcript)

    return output_struct


def build_tree(
    output_struct: dict[str, dict[str, dict[str, list[Transcript]]]],
    outpath: Path,
    max_files: int = 10
) -> Tree:
    tree = Tree(f"[bold blue]{outpath.name}/[/bold blue]")

    for text_type in sorted(output_struct.keys()):
        languages = output_struct[text_type]
        type_branch = tree.add(f"[bold magenta]{text_type}/[/bold magenta]")

        for lang_name in sorted(languages.keys()):
            groups = languages[lang_name]
            lang_branch = type_branch.add(f"[bold green]{lang_name}/[/bold green]")

            for group_name in sorted(groups.keys()):
                transcripts = groups[group_name]
                file_count = len(transcripts)
                group_branch = lang_branch.add(f"[bold cyan]{group_name}/[/bold cyan] ({file_count} files)")

                for t in transcripts[:max_files]:
                    group_branch.add(f"[dim]{t.full_path.name}[/dim]")

                if file_count > max_files:
                    group_branch.add(f"[italic]... and {file_count - max_files} more[/italic]")

    return tree


def copy_files(
    output_struct: dict[str, dict[str, dict[str, list[Transcript]]]],
    outpath: Path
) -> None:
    for text_type, languages in output_struct.items():
        for lang_name, groups in languages.items():
            for group_name, transcripts in groups.items():
                target_dir = outpath / text_type / lang_name / group_name
                target_dir.mkdir(parents=True, exist_ok=True)

                for transcript in transcripts:
                    src = transcript.full_path
                    dst = target_dir / src.name

                    if dst.exists():
                        stem = dst.stem
                        suffix = dst.suffix
                        counter = 1
                        while dst.exists():
                            dst = target_dir / f"{stem}_{counter}{suffix}"
                            counter += 1

                    shutil.copy2(src, dst)

    console.print(f"\n[bold green]Files copied to {outpath}[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize transcripts by language and clinical status.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--i", type=str, required=True, help="Input directory")
    parser.add_argument("--o", type=str, required=True, help="Output directory")
    parser.add_argument("--csv", type=str, required=False, help="CSV with patient_id and clinical_status columns")
    parser.add_argument("--text-type", type=str, required=True, help="Transcript text type")
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    input_path = Path(args.i)
    output_path = Path(args.o)

    csv_path = None
    if args.csv is not None:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            console.print(f"[bold red]Error:[/bold red] CSV path {csv_path} does not exist.")
            return

    output_struct = structure_transcripts(
        dirpath=input_path,
        csv_path=csv_path,
        text_type=args.text_type
    )

    if not output_struct:
        console.print("[yellow]No transcripts found.[/yellow]")
        return

    console.print("[bold]Proposed output structure:[/bold]\n")
    tree = build_tree(output_struct, output_path)
    console.print(tree)

    console.print(f"\n[bold]Target:[/bold] {output_path.resolve()}\n")

    if Confirm.ask("Proceed with copy?"):
        copy_files(output_struct, output_path)
    else:
        console.print("[yellow]Operation cancelled.[/yellow]")


if __name__ == "__main__":
    main()
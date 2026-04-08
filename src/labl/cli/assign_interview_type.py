"""
Decides type of interview for each transcript.
Intended to be used for CONFIRMATION of interview type, rather than final decisions.
Uses vLLM for efficient batch inference.
"""

import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Set GPU before any CUDA imports
if '--gpu' in sys.argv:
    gpu_idx = sys.argv.index('--gpu')
    if gpu_idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[gpu_idx + 1]

import re
import csv
import argparse
from pathlib import Path
from rich.console import Console
from rich.progress import track
from labl.transcripts import Transcript

console = Console()
from vllm import LLM, SamplingParams


def load_model(
    model_name: str = "openai/gpt-oss-120b",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
) -> LLM:
    """
    Load LLM model using vLLM.
    
    Args:
        model_name: HuggingFace model name
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
    
    Returns:
        vLLM LLM instance
    """
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        # max_model_len=80000,
        dtype="auto",
        trust_remote_code=True,
        enable_prefix_caching=True
    )
    
    return llm


def build_messages(
    transcript_content: str,
    thinking: str | None,
) -> list[dict[str, str]]:
    """
    Build the chat messages for interview type classification.
    
    Args:
        transcript_content: Truncated transcript text
        thinking: Optional thinking/reasoning hint
    
    Returns:
        List of message dicts for chat template
    """
    system_prompt = "You are a trained clinical annotator. Only respond in the format prescribed. Keep your reasoning succinct."
    
    user_prompt = f"""Analyze the following transcript excerpt and determine whether it is an OPEN or PSYCHS or UNKNOWN interview.

The OPEN interviews typically:
- Asks questions about the participant's experiences, thoughts, or feelings in a freely flowing format 
- Has no set structure or order

The PSYCHS interviews typically:
- Focuses on clinical symptoms such as anxiety, depression, hallucinations, paranoia, etc.
- Has a linear progression

If neither type of interview is identified, output UNKNOWN. 
- Examples of interviews that are neither might be interviews that relate to cognitive tests, demographic information, etc. 
- The questions will be neither truly open-ended or follow the PSYCHS protocol (a validated psychological scale) 
- Please output UNKNOWN even if you are slightly unsure that it is neither - it is paramount we catch all cases of mis-identified cases

Transcript:
{transcript_content}

Based on the conversation pattern, classify the interview type. Respond with exactly one line in this format:
{{INTERVIEW_TYPE}} where INTERVIEW_TYPE is either OPEN or PSYCHS or UNKNOWN."""

    if thinking is not None:
        system_prompt = f"{system_prompt}\nReasoning approach: {thinking}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_interview_type(response: str) -> str | None:
    """
    Parse interview type from model response.
    
    Returns:
        "OPEN" or "PSYCHS", or None on failure.
    """
    match = re.search(r"\{(OPEN|PSYCHS|UNKNOWN)\}", response, re.IGNORECASE)
    
    if match:
        return match.group(1).upper()
    
    # Fallback: look for standalone OPEN or PSYCHS or UNKNOWN
    match = re.search(r"\b(OPEN|PSYCHS|UNKNOWN)\b", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return None


def classify_interview_type(
    transcript: Transcript,
    llm: LLM,
    sampling_params: SamplingParams,
    thinking: str | None,
    chars: int = 100000,
) -> str | None:
    """
    Classify interview as OPEN or PSYCHS or UNKNOWN.
    
    Args:
        transcript: Transcript object
        llm: vLLM LLM instance
        sampling_params: vLLM sampling parameters
        thinking: Optional thinking/reasoning hint
        chars: Amount of text passed to the model
    
    Returns:
        String with either "OPEN" or "PSYCHS" or "UNKNOWN", or None on failure
    """
    with open(transcript.full_path, "r", encoding="utf-8") as f:
        content = f.read()[:chars]

    messages = build_messages(content, thinking)
    
    outputs = llm.chat(
        messages=[messages],
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    
    response = outputs[0].outputs[0].text.strip()

    interview_type = parse_interview_type(response)

    if interview_type is None:
        console.print(
            f"[yellow]Warning:[/yellow] Failed to parse interview type in {transcript.patient_id} at "
            f"{transcript.filename}: no valid type found in response"
        )
    
    return interview_type


def classify_batch(
    transcripts: list[Transcript],
    llm: LLM,
    sampling_params: SamplingParams,
    thinking: str | None,
    chars: int = 100000,
) -> list[tuple[Transcript, str | None]]:
    """
    Classify interview types for a batch of transcripts.
    
    Args:
        transcripts: List of Transcript objects
        llm: vLLM LLM instance
        sampling_params: vLLM sampling parameters
        thinking: Optional thinking/reasoning hint
        chars: Amount of text passed to the model
    
    Returns:
        List of (transcript, interview_type) tuples
    """
    all_messages = []
    for transcript in transcripts:
        with open(transcript.full_path, "r", encoding="utf-8") as f:
            content = f.read()[:chars]
        messages = build_messages(content, thinking)
        all_messages.append(messages)
    
    outputs = llm.chat(
        messages=all_messages,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    
    results = []
    for transcript, output in zip(transcripts, outputs):
        response = output.outputs[0].text.strip()

        print(response[-300:])

        interview_type = parse_interview_type(response)

        if interview_type is None:
            console.print(
                f"[yellow]Warning:[/yellow] Failed to parse interview type in {transcript.patient_id} at "
                f"{transcript.filename}: no valid type found in response"
            )
        
        results.append((transcript, interview_type))
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decide interview type (OPEN or PSYCHS or UNKNOWN)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--i", type=str, required=True, help="Input directory")
    parser.add_argument("--o", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--thinking",
        type=str,
        required=False,
        default=None,
        help="Thinking parameter for reasoning hint",
    )
    parser.add_argument("--gpu", type=int, required=True, help="GPU device ID")
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size (number of GPUs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (1 for sequential, >1 for batched)",
    )
    args = parser.parse_args()

    input_dir = Path(args.i)
    output_dir = Path(args.o)
    thinking = args.thinking

    output_dir.mkdir(parents=True, exist_ok=True)

    llm = load_model(tensor_parallel_size=args.tp)
    
    sampling_params = SamplingParams(
        max_tokens=250,
        temperature=0.3,  # Low temperature for more deterministic output
    )
    
    Transcript.set_directory_path(input_dir)
    transcripts = list(Transcript.list_transcripts())
    
    failed: list[tuple[str, str]] = []
    open_transcripts: list[dict[str, str]] = []
    psychs_transcripts: list[dict[str, str]] = []
    unknown_transcripts: list[dict[str, str]] = []

    def process_result(transcript: Transcript, interview_type: str | None) -> None:
        """Process a single classification result."""
        if interview_type is None:
            failed.append((str(transcript.filename), "Failed to parse interview type"))
        elif interview_type == "OPEN":
            open_transcripts.append({
                "patient_id": transcript.patient_id,
                "filename": str(transcript.full_path),
            })
        elif interview_type == "PSYCHS":
            psychs_transcripts.append({
                "patient_id": transcript.patient_id,
                "filename": str(transcript.full_path),
            })
        elif interview_type == "UNKNOWN":
            unknown_transcripts.append({
                "patient_id": transcript.patient_id,
                "filename": str(transcript.full_path),
            })
        else:
            failed.append((str(transcript.filename), f"Unknown interview type: {interview_type}"))

    if args.batch_size > 1:
        # Batched processing
        for i in track(range(0, len(transcripts), args.batch_size), description="Classifying interview types..."):
            batch = transcripts[i : i + args.batch_size]
            results = classify_batch(
                transcripts=batch,
                llm=llm,
                sampling_params=sampling_params,
                thinking=thinking,
            )
            
            for transcript, interview_type in results:
                process_result(transcript, interview_type)
    else:
        # Sequential processing
        for transcript in track(transcripts, description="Classifying interview types..."):
            interview_type = classify_interview_type(
                transcript=transcript,
                llm=llm,
                sampling_params=sampling_params,
                thinking=thinking,
            )
            process_result(transcript, interview_type)

    # Write OPEN CSV
    open_csv_path = output_dir / "open_interviews.csv"
    with open(open_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "filename"])
        writer.writeheader()
        writer.writerows(open_transcripts)
    # Write PSYCHS CSV
    psychs_csv_path = output_dir / "psychs_interviews.csv"
    with open(psychs_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "filename"])
        writer.writeheader()
        writer.writerows(psychs_transcripts)
    # Write UNKNOWNS CSV
    unknowns_csv_path = output_dir / "unknown_interviews.csv"
    with open(unknowns_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "filename"])
        writer.writeheader()
        writer.writerows(unknown_transcripts)

    # Report failures
    if failed:
        failed_csv_path = output_dir / "failed_classification.csv"
        with open(failed_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "error"])
            writer.writeheader()
            writer.writerows([{"filename": fn, "error": err} for fn, err in failed])


if __name__ == "__main__":
    main()
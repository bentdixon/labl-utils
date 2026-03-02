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
from utils.transcripts import Transcript
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
    print(f"Loading LLM model: {model_name}...")
    
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=80000,
        dtype="auto",
        trust_remote_code=True,
    )
    
    print(f"Model {model_name} loaded successfully")
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
    
    user_prompt = f"""Analyze the following transcript excerpt and determine whether it is an OPEN or PSYCHS interview.

The OPEN interviews typically:
- Asks questions about the participant's experiences, thoughts, or feelings in a freely flowing format
- Has no set structure or order

The PSYCHS interviews typically:
- Focuses on clinical symptoms such as anxiety, depression, hallucinations, paranoia, etc.
- Has a linear progression

Transcript:
{transcript_content}

Based on the conversation pattern, classify the interview type. Respond with exactly one line in this format:
{{INTERVIEW_TYPE}} where INTERVIEW_TYPE is either OPEN or PSYCHS."""

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
    match = re.search(r"\{(OPEN|PSYCHS)\}", response, re.IGNORECASE)
    
    if match:
        return match.group(1).upper()
    
    # Fallback: look for standalone OPEN or PSYCHS
    match = re.search(r"\b(OPEN|PSYCHS)\b", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return None


def classify_interview_type(
    transcript: Transcript,
    llm: LLM,
    sampling_params: SamplingParams,
    thinking: str | None,
    chars: int = 10000,
) -> str | None:
    """
    Classify interview as OPEN or PSYCHS.
    
    Args:
        transcript: Transcript object
        llm: vLLM LLM instance
        sampling_params: vLLM sampling parameters
        thinking: Optional thinking/reasoning hint
        chars: Amount of text passed to the model
    
    Returns:
        String with either "OPEN" or "PSYCHS", or None on failure
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
    print(f"\n{transcript.filename} --->\n{response}\n")
    
    interview_type = parse_interview_type(response)
    
    if interview_type is None:
        print(
            f"Failed to parse interview type in {transcript.patient_id} at "
            f"{transcript.filename}: no valid type found in response"
        )
    
    return interview_type


def classify_batch(
    transcripts: list[Transcript],
    llm: LLM,
    sampling_params: SamplingParams,
    thinking: str | None,
    chars: int = 10000,
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
        print(f"\n{transcript.filename} --->\n{response}\n")
        
        interview_type = parse_interview_type(response)
        
        if interview_type is None:
            print(
                f"Failed to parse interview type in {transcript.patient_id} at "
                f"{transcript.filename}: no valid type found in response"
            )
        
        results.append((transcript, interview_type))
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decide interview type (OPEN or PSYCHS)",
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
        max_tokens=200,
        temperature=0.0,  # Deterministic output
    )
    
    Transcript.set_directory_path(input_dir)
    transcripts = list(Transcript.list_transcripts())
    
    failed: list[tuple[str, str]] = []
    open_transcripts: list[dict[str, str]] = []
    psychs_transcripts: list[dict[str, str]] = []

    def process_result(transcript: Transcript, interview_type: str | None) -> None:
        """Process a single classification result."""
        print(f"{interview_type} found for {transcript.full_path}")

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
        else:
            failed.append((str(transcript.filename), f"Unknown interview type: {interview_type}"))

    if args.batch_size > 1:
        # Batched processing
        for i in range(0, len(transcripts), args.batch_size):
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
        for transcript in transcripts:
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
    print(f"Wrote {len(open_transcripts)} OPEN interviews to {open_csv_path}")

    # Write PSYCHS CSV
    psychs_csv_path = output_dir / "psychs_interviews.csv"
    with open(psychs_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "filename"])
        writer.writeheader()
        writer.writerows(psychs_transcripts)
    print(f"Wrote {len(psychs_transcripts)} PSYCHS interviews to {psychs_csv_path}")

    # Report failures
    if failed:
        failed_csv_path = output_dir / "failed_classification.csv"
        with open(failed_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "error"])
            writer.writeheader()
            writer.writerows([{"filename": fn, "error": err} for fn, err in failed])
        print(f"Wrote {len(failed)} failed transcripts to {failed_csv_path}")


if __name__ == "__main__":
    main()
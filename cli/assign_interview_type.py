"""
Decides type of interview for each transcript.
Intended to be used for CONFIRMATION of interview type, rather than final decisions.  
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'

import re
import csv
import torch
import argparse
from pathlib import Path
from utils.transcripts import Transcript
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    model_name: str = "openai/gpt-oss-120b",
    device_map: str = "auto",
    attention_impl: str = "kernels-community/vllm-flash-attn3"
):
    """
    Load LLM model and tokenizer.
    Args:
        model_name: HuggingFace model name
        device_map: Device mapping strategy
        attention_impl: Attention implementation to use
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading LLM model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device_map,
        attn_implementation=attention_impl
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model {model_name} loaded successfully")
    return model, tokenizer


def classify_speaker_roles(
    transcript: Transcript,
    model,
    tokenizer,
    thinking: str | None,
    chars: int = 10000,
    temperature: float = 0.2
) -> str | None:
    """
    Classify interview as OPEN or PSYCHS.
    
    Args:
        transcript_sample: A sample of transcript lines
        model: The language model
        tokenizer: The tokenizer
        temperature: Sampling temperature (lower = more deterministic)
    
    Returns:
        String with either "OPEN" or "PSYCHS"
    """
    with open(transcript.full_path, 'r', encoding='utf-8') as f:
        content = f.read()[:chars]


    if thinking is not None:
        prompt = f"""Thinking: {thinking}. Analyze the following transcript excerpt and determine whether it is an OPEN or PSYCHS interview.

    The OPEN interviews typically:
    - Asks questions about the participant's experiences, thoughts, or feelings in a freely flowing format
    - Has no set structure or order

    The PSYCHS interviews typically:
    - Focuses on clinical symptoms such as anxiety, depression, hallucinations, paranoia, etc.
    - Has a linear progression

    Transcript:
    {content}

    Based on the conversation pattern, classify the interview type. Respond with exactly one lines in this format:
    {{INTERVIEW_TYPE}} where INTERVIEW_TYPE is either OPEN or PSYCHS."""
    else:
        prompt = f"""Analyze the following transcript excerpt and determine whether it is an OPEN or PSYCHS interview.

    The OPEN interviews typically:
    - Asks questions about the participant's experiences, thoughts, or feelings in a freely flowing format
    - Has no set structure or order

    The PSYCHS interviews typically:
    - Focuses on clinical symptoms such as anxiety, depression, hallucinations, paranoia, etc.
    - Has a linear progression

    Transcript:
    {content}

    Based on the conversation pattern, classify the interview type. Respond with exactly one lines in this format:
    {{INTERVIEW_TYPE}} where INTERVIEW_TYPE is either OPEN or PSYCHS."""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    try:
        match = re.search(r'\{(OPEN|PSYCHS)\}', response, re.IGNORECASE)
        
        if match:
            interview_type = match.group(1).upper()
            return interview_type
        
        else:
            print(f"Failed to parse interview type in {transcript.patient_id} at {transcript.filename}: no valid type found in response")
            return None
                
    except (AttributeError, ValueError) as e:
        print(f"Failed to parse interview type in {transcript.patient_id} at {transcript.filename} due to error {e}")
        return None

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decide roles in clinicalinterview",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--i", type=str, required=True, help="Input directory")
    parser.add_argument("--o", type=str, required=False, help="Output directory")
    parser.add_argument("--thinking", type=str, required=False, default = None, help="Thinking parameter for GPT-OSS")
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    input_dir = Path(args.i)
    output_dir = Path(args.o) if args.o else None
    thinking = args.thinking

    model, tokenizer = load_model()
    Transcript.set_directory_path(input_dir)

    failed: list[tuple[str, str]] = []
    open_transcripts: list[dict[str, str]] = []
    psychs_transcripts: list[dict[str, str]] = []

    if output_dir is None:
        print("Error: Output directory (--o) is required for CSV output.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for transcript in Transcript.list_transcripts():
        interview_type = classify_speaker_roles(
            transcript=transcript,
            thinking=thinking,
            model=model,
            tokenizer=tokenizer
        )
        print(f"{interview_type} found for {transcript.full_path}")

        if interview_type is None:
            failed.append((str(transcript.filename), "Failed to parse interview type"))
        elif interview_type == "OPEN":
            open_transcripts.append({
                "patient_id": transcript.patient_id,
                "filename": str(transcript.full_path)
            })
        elif interview_type == "PSYCHS":
            psychs_transcripts.append({
                "patient_id": transcript.patient_id,
                "filename": str(transcript.full_path)
            })
        else:
            failed.append((str(transcript.filename), f"Unknown interview type: {interview_type}"))

    # Write OPEN CSV
    open_csv_path = output_dir / "open_interviews.csv"
    with open(open_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "filename"])
        writer.writeheader()
        writer.writerows(open_transcripts)
    print(f"Wrote {len(open_transcripts)} OPEN interviews to {open_csv_path}")

    # Write PSYCHS CSV
    psychs_csv_path = output_dir / "psychs_interviews.csv"
    with open(psychs_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "filename"])
        writer.writeheader()
        writer.writerows(psychs_transcripts)
    print(f"Wrote {len(psychs_transcripts)} PSYCHS interviews to {psychs_csv_path}")

    # Report failures
    if failed:
        failed_csv_path = output_dir / "failed_classification.csv"
        with open(failed_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "error"])
            writer.writeheader()
            writer.writerows([{"filename": fn, "error": err} for fn, err in failed])
        print(f"Wrote {len(failed)} failed transcripts to {failed_csv_path}")


if __name__ == "__main__":
    main()
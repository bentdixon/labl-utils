"""
Decides interviewer / interviewee status for speakers in a transcript.
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


def normalize_speaker_labels(content: str) -> tuple[str, dict[str, str]]:
    """
    Normalize speaker labels to S1, S2, S3 format.
    Handles both standard (S1, S2, S3) and alternative formats (SI, SP, etc.).

    Returns:
        Tuple of (normalized_content, mapping_dict)
        mapping_dict maps normalized labels (S1, S2) back to original labels (SI, SP)
    """
    # Find all unique speaker labels at start of lines
    speaker_pattern = r'^(S[IP\d]+):'
    speakers = set(re.findall(speaker_pattern, content, re.MULTILINE))

    # If all speakers are already in S1, S2, S3 format, no normalization needed
    if all(re.match(r'^S[123]$', s) for s in speakers):
        return content, {}

    # Create mapping: SI -> S1, SP -> S2, or other speakers to S1, S2, S3
    mapping = {}
    reverse_mapping = {}
    normalized_labels = ['S1', 'S2', 'S3']

    # Sort speakers for consistent mapping (SI before SP, etc.)
    sorted_speakers = sorted(speakers)

    for i, original in enumerate(sorted_speakers):
        if i < len(normalized_labels):
            normalized = normalized_labels[i]
            mapping[original] = normalized
            reverse_mapping[normalized] = original

    # Replace speaker labels in content
    normalized_content = content
    for original, normalized in mapping.items():
        # Use word boundary to avoid partial matches
        normalized_content = re.sub(
            rf'^{re.escape(original)}:',
            f'{normalized}:',
            normalized_content,
            flags=re.MULTILINE
        )

    return normalized_content, reverse_mapping


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
    Build the chat messages for speaker role classification.
    
    Args:
        transcript_content: Truncated transcript text
        thinking: Optional thinking/reasoning hint
    
    Returns:
        List of message dicts for chat template
    """
    system_prompt = "You are a trained clinical annotator. Only respond in the format prescribed. Keep your reasoning succinct."
    
    user_prompt = f"""Analyze the following transcript excerpt and determine which speaker is the INTERVIEWER and which is the PARTICIPANT.

The INTERVIEWER typically:
- Asks questions about the participant's experiences, thoughts, or feelings
- Guides the conversation with prompts or follow-up questions
- Uses phrases like "Can you tell me about...", "How did that make you feel?"

The PARTICIPANT typically:
- Responds to questions with personal experiences or opinions
- Provides longer narrative responses
- Shares information about themselves

Transcript:
{transcript_content}

Based on the conversation pattern, classify the speakers. If there are three speakers (S1, S2, S3), only label the INTERVIEWER and PARTICIPANT—leave the third speaker unlabeled.

Respond with exactly two lines:
<speaker>: INTERVIEWER
<speaker>: PARTICIPANT

For example:
S1: INTERVIEWER
S2: PARTICIPANT

Or if S1 is an unlabeled third party:
S2: INTERVIEWER
S3: PARTICIPANT"""

    if thinking is not None:
        system_prompt = f"{system_prompt}\nReasoning approach: {thinking}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_roles(response: str) -> dict[str, str] | None:
    """
    Parse speaker roles from model response.
    
    Returns:
        Dictionary mapping speaker labels (S1/S2/S3) to roles, or None on failure.
        Only contains speakers explicitly labeled as INTERVIEWER or PARTICIPANT.
    """
    roles = {}
    
    # Find all speaker-role assignments (S1, S2, or S3)
    matches = re.findall(r"(S[123]):\s*(INTERVIEWER|PARTICIPANT)", response, re.IGNORECASE)
    
    for speaker, role in matches:
        roles[speaker] = role.upper()
    
    # Validate: must have exactly one INTERVIEWER and one PARTICIPANT
    role_values = list(roles.values())
    if role_values.count("INTERVIEWER") != 1 or role_values.count("PARTICIPANT") != 1:
        return None
    
    if len(roles) != 2:
        return None
    
    return roles


def classify_speaker_roles(
    transcript: Transcript,
    llm: LLM,
    sampling_params: SamplingParams,
    thinking: str | None,
    chars: int = 5000,
) -> tuple[dict[str, str], dict[str, str]] | tuple[None, None]:
    """
    Classify speaker labels as PARTICIPANT or INTERVIEWER.

    Args:
        transcript: Transcript object
        llm: vLLM LLM instance
        sampling_params: vLLM sampling parameters
        thinking: Optional thinking/reasoning hint
        chars: Amount of text passed to the model

    Returns:
        Tuple of (roles_dict, label_mapping) or (None, None) on failure
        roles_dict maps normalized labels (S1, S2) to roles (INTERVIEWER, PARTICIPANT)
        label_mapping maps normalized labels (S1, S2) to original labels (SI, SP)
    """
    with open(transcript.full_path, "r", encoding="utf-8") as f:
        content = f.read()[:chars]

    # Normalize speaker labels
    normalized_content, label_mapping = normalize_speaker_labels(content)

    messages = build_messages(normalized_content, thinking)

    outputs = llm.chat(
        messages=[messages],
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    response = outputs[0].outputs[0].text.strip()
    print(f"\n{transcript.filename} --->\n{response}\n")

    roles = parse_roles(response)

    if roles is None:
        print(
            f"Failed to parse roles in {transcript.patient_id} at "
            f"{transcript.filename} due to failed validation"
        )
        return None, None

    return roles, label_mapping


def classify_batch(
    transcripts: list[Transcript],
    llm: LLM,
    sampling_params: SamplingParams,
    thinking: str | None,
    chars: int = 5000,
) -> list[tuple[Transcript, dict[str, str] | None, dict[str, str]]]:
    """
    Classify speaker roles for a batch of transcripts.

    vLLM efficiently batches multiple requests, so this is more efficient
    than processing one at a time.

    Args:
        transcripts: List of Transcript objects
        llm: vLLM LLM instance
        sampling_params: vLLM sampling parameters
        thinking: Optional thinking/reasoning hint
        chars: Amount of text passed to the model

    Returns:
        List of (transcript, roles, label_mapping) tuples
    """
    all_messages = []
    all_mappings = []

    for transcript in transcripts:
        with open(transcript.full_path, "r", encoding="utf-8") as f:
            content = f.read()[:chars]

        # Normalize speaker labels
        normalized_content, label_mapping = normalize_speaker_labels(content)
        messages = build_messages(normalized_content, thinking)
        all_messages.append(messages)
        all_mappings.append(label_mapping)

    outputs = llm.chat(
        messages=all_messages,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    results = []
    for transcript, output, label_mapping in zip(transcripts, outputs, all_mappings):
        response = output.outputs[0].text.strip()
        print(f"\n{transcript.filename} --->\n{response}\n")

        roles = parse_roles(response)

        if roles is None:
            print(
                f"Failed to parse roles in {transcript.patient_id} at "
                f"{transcript.filename} due to failed validation"
            )

        results.append((transcript, roles, label_mapping))

    return results


def _write_output(
    transcript: Transcript,
    roles: dict[str, str],
    label_mapping: dict[str, str],
    input_dir: Path,
    output_dir: Path | None,
) -> None:
    """
    Write the transcript with replaced speaker labels.

    Args:
        transcript: Transcript object
        roles: Dict mapping normalized labels (S1, S2) to roles (INTERVIEWER, PARTICIPANT)
        label_mapping: Dict mapping normalized labels (S1, S2) to original labels (SI, SP)
        input_dir: Input directory path
        output_dir: Optional output directory path
    """
    with open(transcript.full_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Map normalized labels to roles, then replace original labels with roles
    for normalized_label, role in roles.items():
        # Get the original label (SI, SP, etc.) or use normalized if no mapping
        original_label = label_mapping.get(normalized_label, normalized_label)
        content = re.sub(
            rf"^{re.escape(original_label)}:",
            f"{role}:",
            content,
            flags=re.MULTILINE
        )

    if output_dir:
        relative_path = transcript.full_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = transcript.full_path

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decide roles in clinical interview",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--i", type=str, required=True, help="Input directory")
    parser.add_argument("--o", type=str, required=False, help="Output directory")
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
    output_dir = Path(args.o) if args.o else None
    thinking = args.thinking

    llm = load_model(tensor_parallel_size=args.tp)
    
    sampling_params = SamplingParams(
        max_tokens=700,
        temperature=0.0,  # Deterministic output
    )
    
    Transcript.set_directory_path(input_dir)
    transcripts = list(Transcript.list_transcripts())
    
    failed = []

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

            for transcript, roles, label_mapping in results:
                print(f"{roles} found for {transcript.filename}")
                if roles is None:
                    failed.append(transcript.filename)
                else:
                    _write_output(transcript, roles, label_mapping, input_dir, output_dir)
    else:
        # Sequential processing (original behavior)
        for transcript in transcripts:
            roles, label_mapping = classify_speaker_roles(
                transcript=transcript,
                llm=llm,
                sampling_params=sampling_params,
                thinking=thinking,
            )
            print(f"{roles} found for {transcript.filename}")
            if roles is None:
                failed.append(transcript.filename)
            else:
                _write_output(transcript, roles, label_mapping, input_dir, output_dir)

    if failed:
        log_dir = output_dir if output_dir else input_dir
        output_path = log_dir / "logs" / "failed_files.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"The following files could not be parsed: {failed}")

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(["File"])
            for file in failed:
                writer.writerow([file])


if __name__ == "__main__":
    main()
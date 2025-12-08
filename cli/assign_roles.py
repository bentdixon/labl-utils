"""
Decides interviewer / interviewee status for speakers in a transcript.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'

import re
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
    
    print(f"Model loaded successfully")
    return model, tokenizer


def classify_speaker_roles(
    transcript: Transcript,
    model,
    tokenizer,
    thinking: str | None,
    chars: int = 10000,
    temperature: float = 0.2
) -> dict[str, str] | None:
    """
    Classify S1 and S2 speaker labels as PARTICIPANT or INTERVIEWER.
    
    Args:
        transcript_sample: A sample of transcript lines containing S1/S2 labels
        model: The language model
        tokenizer: The tokenizer
        temperature: Sampling temperature (lower = more deterministic)
    
    Returns:
        Dictionary mapping speaker labels to roles, e.g., {"S1": "INTERVIEWER", "S2": "PARTICIPANT"}
    """
    with open(transcript.full_path, 'r', encoding='utf-8') as f:
        content = f.read()[:chars]


    if thinking is not None:
        prompt = f"""Thinking: {thinking}. Analyze the following transcript excerpt and determine which speaker is the INTERVIEWER and which is the PARTICIPANT.

    The INTERVIEWER typically:
    - Asks questions about the participant's experiences, thoughts, or feelings
    - Guides the conversation with prompts or follow-up questions
    - Uses phrases like "Can you tell me about...", "How did that make you feel?"

    The PARTICIPANT typically:
    - Responds to questions with personal experiences or opinions
    - Provides longer narrative responses
    - Shares information about themselves

    Transcript:
    {content}

    Based on the conversation pattern, classify each speaker. Respond with exactly two lines in this format:
    S1: INTERVIEWER or S1: PARTICIPANT
    S2: INTERVIEWER or S2: PARTICIPANT"""
    else:
        prompt = f"""Analyze the following transcript excerpt and determine which speaker is the INTERVIEWER and which is the PARTICIPANT.

    The INTERVIEWER typically:
    - Asks questions about the participant's experiences, thoughts, or feelings
    - Guides the conversation with prompts or follow-up questions
    - Uses phrases like "Can you tell me about...", "How did that make you feel?"

    The PARTICIPANT typically:
    - Responds to questions with personal experiences or opinions
    - Provides longer narrative responses
    - Shares information about themselves

    Transcript:
    {content}

    Based on the conversation pattern, classify each speaker. Respond with exactly two lines in this format:
    S1: INTERVIEWER or S1: PARTICIPANT
    S2: INTERVIEWER or S2: PARTICIPANT"""

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
    
    roles = {}
    try:
        s1_match = re.search(r'S1:\s*(INTERVIEWER|PARTICIPANT)', response, re.IGNORECASE)
        s2_match = re.search(r'S2:\s*(INTERVIEWER|PARTICIPANT)', response, re.IGNORECASE)
        
        if s1_match:
            roles["S1"] = s1_match.group(1).upper()
        if s2_match:
            roles["S2"] = s2_match.group(1).upper()
        
        # Validate: should have exactly one of each role
        if set(roles.values()) != {"INTERVIEWER", "PARTICIPANT"}:
            print(f"Failed to parse roles in {transcript.patient_id} at {transcript.filename} due to failed validation for one of each role: {roles.values()}")
            return None
            
    except (AttributeError, ValueError) as e:
        print(f"Failed to parse roles in {transcript.patient_id} at {transcript.filename} due to error {e}")
        return None
    
    return roles


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

    failed = []

    for transcript in Transcript.list_transcripts():
        roles = classify_speaker_roles(transcript = transcript, thinking = thinking, model = model, tokenizer = tokenizer)
        if roles is None:
            failed.append(transcript.filename)
        else:
            with open(transcript.full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            for speaker, role in roles.items():
                content = re.sub(rf'^{speaker}:', f'{role}:', content, flags=re.MULTILINE)

            if args.o:
                relative_path = transcript.full_path.relative_to(input_dir)
                output_path = output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_path = transcript.full_path

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            



if __name__ == "__main__":
    main()
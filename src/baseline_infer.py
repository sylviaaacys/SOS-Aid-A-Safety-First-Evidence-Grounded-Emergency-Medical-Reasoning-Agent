#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "epfl-llm/meditron-7b"  # change if needed
DTYPE = torch.bfloat16                 # or torch.float16
DEVICE_MAP = "auto"

PROMPT =[
"Someone collapses and is not breathing", "Someone is drowning, what can i do?", 
"My friend is choking and cant speak, how can I help them?",
"A child fell and is bleeding a lot from the head, what should I do?",
"Someone was electrocuted and is unconscious, what are the first steps?"
]

'''
MODELS_LOCAL: Dict[str, str] = {
        "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "meditron-7b": "epfl-llm/meditron-7b",
}
'''

# ----------------------------
# Load tokenizer & model
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=False,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=DTYPE,
    device_map=DEVICE_MAP,
    local_files_only=True,
)

model.eval()

# Ensure pad token exists
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------
# Tokenize (single prompt)
# ----------------------------
for p in PROMPT:
    inputs = tokenizer(
        p,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # ----------------------------
    # Generate
    # ----------------------------
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,            # deterministic
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # ----------------------------
    # Decode ONLY generated text
    # ----------------------------
    gen_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
    output = tokenizer.decode(gen_ids, skip_special_tokens=True)

    print("\n=== MODEL OUTPUT ===\n")
    print(output.strip())
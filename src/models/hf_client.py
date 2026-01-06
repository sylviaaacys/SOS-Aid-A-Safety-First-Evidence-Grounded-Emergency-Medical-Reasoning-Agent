from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Cache per model_id (important if you evaluate multiple HF models)
_MODEL_CACHE: Dict[str, AutoModelForCausalLM] = {}
_TOKENIZER_CACHE: Dict[str, AutoTokenizer] = {}

def get_local_hf_model(model_id: str, dtype: torch.dtype = torch.bfloat16) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if model_id in _MODEL_CACHE and model_id in _TOKENIZER_CACHE:
        return _MODEL_CACHE[model_id], _TOKENIZER_CACHE[model_id]

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        torch_dtype=dtype,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    _MODEL_CACHE[model_id] = model
    _TOKENIZER_CACHE[model_id] = tokenizer
    return model, tokenizer


@dataclass
class LocalHFChatClient:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    max_new_tokens: int = 100
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False

    @torch.no_grad()
    def generate(self, prompt: str, system_prompt: str) -> str:
        # Works even if tokenizer.chat_template missing
        full_prompt = (
            "### System:\n"
            f"{system_prompt}\n\n"
            "### User:\n"
            f"{prompt}\n\n"
            "### Assistant:\n"
        )
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_p=self.top_p if self.do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

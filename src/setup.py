# models.py
import os
import json
import torch
from dataclasses import dataclass
from typing import Optional, Any, Dict, List

from config import setup_hf_cache_offline, load_api_config
from models import Deps
from models.hf_client import get_local_hf_model, LocalHFChatClient
from models.api_client import OpenAICompatibleClient
from retriever import load_retriever
from tavily import TavilyClient

'''
MODELS_LOCAL: Dict[str, str] = {
        "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "meditron-7b": "epfl-llm/meditron-7b",
}
'''

#export TAVILY_API_KEY="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"
model_id = "epfl-llm/meditron-7b"
#"mistralai/Mixtral-8x7B-Instruct-v0.1"#api 
MODEL_ID = None #api
classifier_llm_hf = True
answer_llm_hf= True

def load_deps() -> Deps:
    # Create your OpenAI client, HF pipeline
    model, tokenizer = get_local_hf_model(model_id)
     # --- Local HF clients -------
    local_classifier = LocalHFChatClient(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=120,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
    )

    local_answer = LocalHFChatClient(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
    )

    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    def tavily_search(query: str, max_results: int = 5):
        res = tavily_client.search(query=query, max_results=max_results)
        return res.get("results", [])

    if classifier_llm_hf:
        classifier_llm = local_classifier
    else:
        # API client
        api_cfg = load_api_config(model=MODEL_ID)
        classifier_llm = OpenAICompatibleClient(model=api_cfg.model, api_key=api_cfg.api_key, base_url=api_cfg.base_url)
        
    if answer_llm_hf:
        answer_llm = local_answer
    else:
        api_cfg = load_api_config(model=MODEL_ID)
        answer_llm = OpenAICompatibleClient(model=MODEL_ID, api_key=API_KEY, base_url=BASE_URL)

    # --- Create a retriever ---
    retriever = load_retriever()
    deps = Deps(
        classifier_llm=classifier_llm,
        answer_llm=answer_llm,
        retriever=retriever,
        tavily_search=tavily_search,   
    )
    return deps
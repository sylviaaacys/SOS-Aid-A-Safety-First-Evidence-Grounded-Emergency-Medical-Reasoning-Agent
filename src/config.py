import os
from dataclasses import dataclass
from typing import Optional

def setup_hf_cache_offline(
    hf_home: str = "/data1/baseline-models",
    hub_cache: str = "/data1/baseline-models/hub",
    offline: bool = True,
) -> None:
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", hub_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hub_cache)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

@dataclass(frozen=True)
class APIConfig:
    api_key: str
    base_url: str
    model: str

@dataclass(frozen=True)
class TavilyConfig:
    api_key: str

def load_api_config(
    model: Optional[str] = None,
    api_key_env: str = "API_KEY",
    base_url_env: str = "BASE_URL",
    default_model: str = "Qwen2.5-72B-Instruct",
    default_base_url: str = "https://api-inference.modelscope.cn/v1",
    tavily_api: str = "TAVILY_API_KEY"
) -> APIConfig:
    api_key = os.getenv(api_key_env, "")
    base_url = os.getenv(base_url_env, default_base_url)
    tavily_api = os.getenv(tavily_api, "")
    if not tavily_api:
        raise ValueError(f"Missing tavily api key var {env}.")
    if not api_key:
        raise ValueError(f"Missing env var {api_key_env}. Do NOT hardcode keys in code.")
    return APIConfig(api_key=api_key, base_url=base_url, model=model, tavily_api=tavily_api or default_model)
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class OpenAICompatibleClient:
    model: str
    api_key: str
    base_url: str

    def __post_init__(self):
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, system_prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return resp.choices[0].message.content

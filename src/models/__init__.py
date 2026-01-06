from dataclasses import dataclass
from typing import Any, Optional, Callable, List, Dict

@dataclass
class Deps:
    classifier_llm: Any
    answer_llm: Any
    retriever: Optional[Any] = None
    tavily_search: Optional[Callable[[str, int], List[Dict]]] = None
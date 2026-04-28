from data import QueryState, TokenMapping
from typing import List, Dict, Any
from abc import ABC, abstractmethod


class DomainClassifier(ABC):
    @abstractmethod
    def process(self, state: QueryState) -> str:
        pass

class IntentClassifier(ABC):
    @abstractmethod
    def process(self, state: QueryState) -> str:
        pass

class LabelGenerator(ABC):
    @abstractmethod
    def process(self, state: QueryState) -> List[str]:
        # This is where your future RAG implementation will go.
        # It can pull documents based on state.domain and state.intent.
        pass

class EntityExtractor(ABC):
    @abstractmethod
    def process(self, state: QueryState) -> List[Dict[str, Any]]:
        # NER model or regex goes here. Only looks for state.potential_labels.
        pass

class PolicyEvaluator(ABC):
    @abstractmethod
    def process(self, state: QueryState) -> List[TokenMapping]:
        # Evaluates extracted_entities and decides: drop, abstract, or keep.
        pass

class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        # General LLM wrapper (OpenAI, local Llama, etc.)
        pass
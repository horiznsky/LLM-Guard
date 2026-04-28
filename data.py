from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class TokenMapping:
    original_token: str
    replacement_token: str  # e.g., "[CITY_1]", "Fakeville", or ""
    label: str
    action: str             # 'drop', 'fake', 'abstract', 'keep'

@dataclass
class QueryState:
    query_id: str
    raw_query: str
    domain: str = None
    intent: str = None
    potential_labels: List[str] = field(default_factory=list)
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list) # e.g., [{'token': 'Paris', 'label': 'LOC'}]
    mappings: List[TokenMapping] = field(default_factory=list)
    sanitized_query: str = None
    llm_sanitized_response: str = None
    final_restored_response: str = None
    llm_raw_response: str = None # Specifically for your testing requirement
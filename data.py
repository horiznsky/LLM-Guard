from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class TokenMapping:
    original_token: str
    replacement_token: str  
    label: str
    action: str             
    start: int = -1         # <-- ADDED for exact string slicing
    end: int = -1           # <-- ADDED for exact string slicing

@dataclass
class QueryState:
    query_id: str
    raw_query: str
    domain: str = None
    intent: str = None
    potential_labels: Dict[str, str] = field(default_factory=dict) # <-- CHANGED to Dict for GLiNER descriptions
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list) 
    mappings: List[TokenMapping] = field(default_factory=list)
    sanitized_query: str = None
    llm_sanitized_response: str = None
    final_restored_response: str = None
    llm_raw_response: str = None
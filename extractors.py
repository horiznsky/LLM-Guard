from typing import List, Dict, Any
from gliner2 import GLiNER2
from interfaces import EntityExtractor, QueryState

class GlinerExtractor(EntityExtractor):
    def __init__(self, model_name="fastino/gliner2-base-v1"):
        self.model = GLiNER2.from_pretrained(model_name)
        
        # Hardcode the most common sensitive labels so we never miss them
        self.base_labels = [
            "person", "name", "address", "phone number", "email", 
            "credit card", "account number", "medical condition", 
            "medication", "disease", "api key", "password", "age"
        ]

    def process(self, state: QueryState) -> List[Dict[str, Any]]:
        # Combine the LLM's dynamic labels with our guaranteed base labels
        combined_labels = list(set(self.base_labels + [L.lower() for L in state.potential_labels]))
            
        result = self.model.extract_entities(
            state.raw_query, 
            combined_labels,
            include_confidence=True
        )
        
        extracted = []
        for label, entities_list in result.get("entities", {}).items():
            for e in entities_list:
                extracted.append({
                    "token": e["text"],
                    "label": label,
                    "score": e.get("confidence", 0.0)
                })
                
        return extracted
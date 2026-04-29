from typing import List, Dict, Any
from gliner2 import GLiNER2
from interfaces import EntityExtractor, QueryState

class GlinerExtractor(EntityExtractor):
    def __init__(self, model_name="fastino/gliner2-base-v1"):
        self.model = GLiNER2.from_pretrained(model_name)
        
        # Hardcode with rich descriptions for higher accuracy
        self.base_labels = {
            "person": "First, last, or full names of individuals",
            "address": "Physical street addresses, cities, states, or zip codes",
            "phone number": "Telephone or mobile numbers",
            "email": "Email addresses",
            "credit card": "Credit card numbers",
            "account number": "Bank, financial, or user account numbers",
            "medical condition": "Diseases, illnesses, or medical symptoms",
            "medication": "Names of drugs, pharmaceuticals, or medications",
            "api key": "Alphanumeric API keys, tokens, or passwords",
            "age": "A person's age in years or months",
            "social security number": "9-digit SSN or social security identifiers",
            "money": "Monetary values, amounts, or salaries"
        }

    def process(self, state: QueryState) -> List[Dict[str, Any]]:
        # Combine base dictionary with the LLM's dynamic dictionary
        combined_labels = self.base_labels.copy()
        
        if isinstance(state.potential_labels, dict):
            for k, v in state.potential_labels.items():
                combined_labels[k.lower()] = v
        elif isinstance(state.potential_labels, list):
            # Fallback if the LLM messes up and returns a list
            for label in state.potential_labels:
                combined_labels[label.lower()] = f"Entities related to {label}"
            
        # Use GLiNER 2's advanced features
        result = self.model.extract_entities(
            state.raw_query, 
            combined_labels,
            include_confidence=True,
            include_spans=True
        )
        
        extracted = []
        for label, entities_list in result.get("entities", {}).items():
            for e in entities_list:
                # Optional: Filter out super low confidence extractions here
                if e.get("confidence", 0.0) > 0.3:
                    extracted.append({
                        "token": e["text"],
                        "label": label,
                        "score": e.get("confidence", 0.0),
                        "start": e.get("start", 0),
                        "end": e.get("end", 0)
                    })
                
        return extracted
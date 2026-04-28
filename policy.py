import json
from typing import List
from openai import OpenAI
from groq import Groq
from interfaces import PolicyEvaluator, QueryState, TokenMapping
from config import GROQ_API_KEY

class LLMPolicyEvaluator(PolicyEvaluator):
    # Defaulting to your requested local 3B model
    def __init__(self, model_name="qwen2.5:3b"):
        self.model_name = model_name
        self.is_local = ":" in model_name or "qwen" in model_name.lower()

        if self.is_local:
            print(f"🔌 Policy Evaluator: Initialized LOCAL model ({self.model_name})")
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        else:
            print(f"☁️ Policy Evaluator: Initialized CLOUD API ({self.model_name})")
            self.client = Groq(api_key=GROQ_API_KEY)

    def process(self, state: QueryState) -> List[TokenMapping]:
        if not state.extracted_entities:
            return []

        # Extract just the string tokens for the prompt
        entities_json = json.dumps([e['token'] for e in state.extracted_entities])
        
        prompt = f"""
        Query: {state.raw_query}
        Intent: {state.intent}
        Extracted Entities: {entities_json}
        
        Evaluate each entity against the intent. Classify the action required for privacy:
        - "drop": Highly sensitive, not needed for the intent. Remove entirely.
        - "abstract": Sensitive, but the category is needed for context. Replace with label (e.g., Paris -> [LOCATION]).
        - "keep": Core to the task (T_core). If removed, the intent fails.
        
        Output valid JSON with a single key "mappings", containing a list of objects with "original_token", "action", and "replacement_token".
        If dropping, replacement_token is "". If keeping, replacement_token is the original token. If abstracting, use the label in brackets.
        """

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a strict privacy policy engine. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            mappings = []
            
            # Match back to the original labels from GLiNER
            for mapped_item in result.get("mappings", []):
                original = mapped_item["original_token"]
                label = next((e["label"] for e in state.extracted_entities if e["token"] == original), "UNKNOWN")
                
                mappings.append(TokenMapping(
                    original_token=original,
                    replacement_token=mapped_item["replacement_token"],
                    label=label,
                    action=mapped_item["action"]
                ))
            return mappings
            
        except Exception as e:
            print(f"Policy Evaluation Error ({self.model_name}): {e}")
            return []
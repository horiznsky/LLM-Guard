import json
from typing import List
from openai import OpenAI
from groq import Groq
from interfaces import PolicyEvaluator, QueryState, TokenMapping
from config import GROQ_API_KEY, RUNPOD_OLLAMA_URL

class LLMPolicyEvaluator(PolicyEvaluator):
    # Recommend using qwen2.5:7b for strict policy logic
    def __init__(self, model_name="qwen2.5:7b"):
        self.model_name = model_name
        self.is_runpod = True

        if self.is_runpod:
            print(f"🛡️ Policy Evaluator: Initialized RUNPOD ({self.model_name})")
            self.client = OpenAI(
                base_url=RUNPOD_OLLAMA_URL, # Uses the proxy URL from config.py
                api_key="runpod_ollama"     # Dummy key, required by library
            )
        else:
            print(f"☁️ Policy Evaluator: Initialized CLOUD API ({self.model_name})")
            self.client = Groq(api_key=GROQ_API_KEY)

    def process(self, state: QueryState) -> List[TokenMapping]:
        if not state.extracted_entities:
            return []

        # Pass both token and label to the LLM so it knows exactly what the entity represents
        entities_for_prompt = [{"token": e['token'], "label": e['label']} for e in state.extracted_entities]
        entities_json = json.dumps(entities_for_prompt)
        
        # STRONGER PROMPT: Addressed the high "Undermasking Rate" (FN)
        prompt = f"""
        Query: '{state.raw_query}'
        Intent: '{state.intent}'
        Extracted Entities: {entities_json}
        
        Evaluate each entity against the intent. You are a strict privacy policy engine applying the Rule of Least Privilege.
        Mask everything unless it mathematically breaks the task.
        
        Classify the action required for privacy:
        - "drop": Highly sensitive (Names, SSNs, exact addresses, IDs). Not needed to answer the intent. Remove entirely.
        - "abstract": Sensitive, but the category is needed for context. Replace with the label in brackets (e.g., Paris -> [LOCATION], 500mg -> [DOSAGE]).
        - "keep": Core to the task (T_core). ONLY use this if the token is the explicit target of the user's functional request (e.g., "account balance" in "check_account_balance"). NEVER keep PII/PHI.
        
        Output valid JSON with a single key "mappings", containing a list of objects with "original_token", "action", and "replacement_token".
        If dropping, replacement_token is "". If keeping, replacement_token is the original token. If abstracting, use the label in brackets.
        """

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a strict zero-trust privacy policy engine. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            mappings = []
            # Match back to the original labels AND spans from GLiNER extraction
            for mapped_item in result.get("mappings", []):
                # Use .get() with fallbacks in case the 3B model forgets a key
                original = mapped_item.get("original_token", "")
                if not original:
                    continue # Skip if the LLM hallucinated an empty item
                
                label = "UNKNOWN"
                start = -1
                end = -1
                
                # Find the corresponding entity to grab its span
                for e in state.extracted_entities:
                    if e["token"] == original:
                        label = e.get("label", "UNKNOWN")
                        start = e.get("start", -1)
                        end = e.get("end", -1)
                        break
                
                mappings.append(TokenMapping(
                    original_token=original,
                    # Safely default to "" if missing
                    replacement_token=mapped_item.get("replacement_token", ""),
                    label=label,
                    # Safely default to "keep" to prevent accidental data loss if it bugs out
                    action=mapped_item.get("action", "keep").lower(), 
                    start=start,
                    end=end
                ))
            return mappings
            
        except Exception as e:
            print(f"Policy Evaluation Error ({self.model_name}): {e}")
            return []
import json
import re
from typing import List
from openai import OpenAI
from groq import Groq
from interfaces import PolicyEvaluator, QueryState, TokenMapping
from config import GROQ_API_KEY, RUNPOD_OLLAMA_URL

class LLMPolicyEvaluator(PolicyEvaluator):
    def __init__(self, model_name="qwen2.5:7b"):
        self.model_name = model_name
        self.is_runpod = True

        if self.is_runpod:
            print(f"🛡️ Policy Evaluator: Initialized RUNPOD ({self.model_name})")
            self.client = OpenAI(
                base_url=RUNPOD_OLLAMA_URL, 
                api_key="runpod_ollama"     
            )
        else:
            print(f"☁️ Policy Evaluator: Initialized CLOUD API ({self.model_name})")
            self.client = Groq(api_key=GROQ_API_KEY)

    def process(self, state: QueryState) -> List[TokenMapping]:
        if not state.extracted_entities:
            return []

        entities_for_prompt = [{"token": e['token'], "label": e['label']} for e in state.extracted_entities]
        entities_json = json.dumps(entities_for_prompt)
        
        # PROMPT RETAINED: Full instructions kept to maintain high-quality decision making.
        prompt = f"""
        Query: "{state.raw_query}"
        Domain: "{state.domain}"
        Intent: "{state.intent}"
        Entities: {entities_json}

        You are a Senior Privacy Architect specializing in Zero-Trust LLM Gateways.
        Your task is to sanitize the query based on the extraction list.

        THE THREE COMMANDMENTS OF CLASSIFICATION:
        For EACH entity provided in the Entities list, you MUST assign ONE of the following actions:

        1. "drop" (Zero Utility): 
        - Definition: Direct Identifiers like SSNs, phone numbers, exact building addresses, or full personal names.
        - Rule: These add NO VALUE to the query's intent fulfillment and represent a high privacy risk. Drop them entirely.

        2. "abstract" (Structural Necessity): 
        - Definition: PII that is a privacy risk but is essential for sentence structure or context. 
        - Rule: Replace with its categorical label (e.g., [CITY]). Prefer this over dropping if removing it makes the sentence nonsensical.

        3. "keep" (Mission Critical): 
        - Definition: Strings mechanically essential to fulfilling the Intent. 
        - Rule: If this specific token is removed, the task becomes impossible. NEVER use this for direct human identifiers.

        You MUST output ONLY valid JSON matching this exact structure:
        {{
        "mappings": [
            {{"original_token": "...", "action": "drop|abstract|keep"}}
        ]
        }}
        """
        try:
            # TURBO: Using response_format ensures valid JSON without reasoning overhead
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a privacy architect. You always respond in JSON."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            result = json.loads(result_text)

            mappings = []
            
            for mapped_item in result.get("mappings", []):
                original = mapped_item.get("original_token", "")
                if not original: continue 
                
                label = "UNKNOWN"
                start = -1
                end = -1
                
                # 1. Grab the label and coordinates from GLiNER's extraction
                for e in state.extracted_entities:
                    if e["token"] == original:
                        label = e.get("label", "UNKNOWN")
                        start = e.get("start", -1)
                        end = e.get("end", -1)
                        break
                
                # 2. AUTO-ASSIGN REPLACEMENT TOKEN
                action = mapped_item.get("action", "keep").lower()
                if action == "drop":
                    replacement_token = ""
                elif action == "abstract":
                    replacement_token = f"[{label.upper()}]"
                else: # action is 'keep'
                    replacement_token = original
                
                mappings.append(TokenMapping(
                    original_token=original,
                    replacement_token=replacement_token,
                    label=label,
                    action=action, 
                    start=start,
                    end=end
                ))
            return mappings
            
        except Exception as e:
            print(f"Policy Evaluation Error ({self.model_name}): {e}")
            return []
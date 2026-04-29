import json
from openai import OpenAI
from groq import Groq
from typing import List
from interfaces import DomainClassifier, IntentClassifier, LabelGenerator, QueryState
from config import GROQ_API_KEY, RUNPOD_OLLAMA_URL

class FastRoutingClassifier(DomainClassifier, IntentClassifier, LabelGenerator):
    # Recommend using llama3.1:8b for routing
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name
        
        # Toggle to True to use your RunPod models
        self.is_runpod = True

        if self.is_runpod:
            print(f"🚀 Routing Classifier: Initialized RUNPOD ({self.model_name})")
            self.client = OpenAI(
                base_url=RUNPOD_OLLAMA_URL, # Uses the proxy URL from config.py
                api_key="runpod_ollama"     # Dummy key, required by library
            )
        else:
            print(f"☁️ Routing Classifier: Initialized CLOUD API ({self.model_name})")
            self.client = Groq(api_key=GROQ_API_KEY)

    def process_domain(self, state: QueryState) -> str:
        # Give it strict choices based on your dataset
        prompt = f"""
        Analyze this query: '{state.raw_query}'. 
        Choose the domain ONLY from this list: [Healthcare, Finance, Software Engineering].
        Output a JSON object with a single key 'domain'.
        """
        return self._call_json(prompt).get("domain", "general")

    def process_intent(self, state: QueryState) -> str:
        # Give it examples of the EXACT snake_case format your CSV expects
        prompt = f"""
        Query: '{state.raw_query}'
        Domain: '{state.domain}'
        Determine the intent. 
        CRITICAL: Output ONLY from typical intents like: 'check_medication_interaction', 'check_account_balance', 'fix_api_error', 'apply_for_loan', 'symptom_diagnosis'.
        Output a JSON object with a single key 'intent'.
        """
        return self._call_json(prompt).get("intent", "unknown_task")
    
    def process_labels(self, state: QueryState) -> dict:
        prompt = f"""
        Query: '{state.raw_query}'
        What sensitive entity categories are present in this text? 
        Output a JSON object with a key 'labels' containing a dictionary mapping label names to brief descriptions.
        Include standard categories if present.
        Example: {{"labels": {{"PERSON": "Names of individuals", "AGE": "A person's age in years", "DISEASE": "Medical conditions or symptoms"}}}}
        """
        # Returns a dict like {"PERSON": "...", "DISEASE": "..."}
        return self._call_json(prompt).get("labels", {})
    
    def _call_json(self, prompt: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a data routing API. You must output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"LLM Routing Error ({self.model_name}): {e}")
            return {}

    def process(self, state: QueryState):
        pass
import json
from openai import OpenAI
from groq import Groq
from typing import List
from interfaces import DomainClassifier, IntentClassifier, LabelGenerator, QueryState
from config import GROQ_API_KEY

class FastRoutingClassifier(DomainClassifier, IntentClassifier, LabelGenerator):
    # Defaulting to your requested local 3B model
    def __init__(self, model_name="qwen2.5:3b"):
        self.model_name = model_name
        
        # Auto-detect if we should use local Ollama or cloud API
        self.is_local = ":" in model_name or "qwen" in model_name.lower()

        if self.is_local:
            print(f"🔌 Routing Classifier: Initialized LOCAL model ({self.model_name})")
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama" 
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
    
    def process_labels(self, state: QueryState) -> List[str]:
        prompt = f"""
        Query: '{state.raw_query}'
        What entity labels are present in this text? 
        Output a JSON object with a key 'labels' containing a list of strings.
        Include standard categories like: "PERSON", "AGE", "ADDRESS", "ACCOUNT_NUMBER", "DISEASE", "MEDICATION", "MONEY", "CREDIT_SCORE", "API_KEY", "SYMPTOM".
        Add any other specific labels relevant to the text.
        """
        return self._call_json(prompt).get("labels", [])

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
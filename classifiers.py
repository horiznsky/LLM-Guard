# import json
# from openai import OpenAI
# from groq import Groq
# from typing import List
# from interfaces import DomainClassifier, IntentClassifier, LabelGenerator, QueryState
# from config import GROQ_API_KEY, RUNPOD_OLLAMA_URL

# class FastRoutingClassifier(DomainClassifier, IntentClassifier, LabelGenerator):
#     # Recommend using llama3.1:8b for routing
#     def __init__(self, model_name="qwen2.5:7b"):
#         self.model_name = model_name
        
#         # Toggle to True to use your RunPod models
#         self.is_runpod = True

#         if self.is_runpod:
#             print(f"🚀 Routing Classifier: Initialized RUNPOD ({self.model_name})")
#             self.client = OpenAI(
#                 base_url=RUNPOD_OLLAMA_URL, # Uses the proxy URL from config.py
#                 api_key="runpod_ollama"     # Dummy key, required by library
#             )
#         else:
#             print(f"☁️ Routing Classifier: Initialized CLOUD API ({self.model_name})")
#             self.client = Groq(api_key=GROQ_API_KEY)

#     def process_domain(self, state: QueryState) -> str:
#         # Removed the strict hardcoded choices. Let the LLM infer the context.
#         prompt = f"""
#         Analyze this query: '{state.raw_query}'. 
#         Determine the broad industry, field, or domain this query belongs to (e.g., healthcare, finance, education, retail, software engineering,law, general conversation).
#         Output the domain as a concise, descriptive lowercase string.
#         Output a JSON object with a single key 'domain'.
#         """
#         return self._call_json(prompt).get("domain", "general").lower()

#     # In classifiers.py
#     def process_intent(self, state: QueryState) -> str:
#         prompt = f"""
#         Query: '{state.raw_query}'
#         Domain: '{state.domain}'
        
#         Determine the primary intent or task of the user's query. 
#         Output the intent as a concise verb-noun phrase in snake_case. 
        
#         PATTERN EXAMPLES:
#         Query: "What happens if I take Aspirin with my current meds?" -> {{"intent": "check_medication_interaction"}}
#         Query: "How much money is in my savings?" -> {{"intent": "check_account_balance"}}
#         Query: "Why is my Python script throwing a 500 error?" -> {{"intent": "debug_code"}}
#         Query: "Book a flight to Paris for tomorrow." -> {{"intent": "book_flight"}}
        
#         Output a JSON object with a single key 'intent' based on the query above.
#         """
#         return self._call_json(prompt).get("intent", "unknown_task")
    
#     def process_labels(self, state: QueryState) -> dict:
#         prompt = f"""
#         Query: '{state.raw_query}'
#         Domain: '{state.domain}'
        
#         Identify all potential sensitive, confidential, or PII entity categories present in this text.
#         You must look for BOTH Universal identifiers AND Domain-Specific sensitive data.
        
#         Extraction Rules:
#         1. Universal Baseline: ALWAYS identify standard direct identifiers regardless of domain (e.g., exact names, phone numbers, emails, street addresses, government IDs, financial data).
#         2. Domain Specificity: ALSO identify entities that are sensitive specifically within the '{state.domain}' domain (e.g., API keys, medical diagnoses, proprietary project names).
#         3. Extreme Granularity: Do not use broad categories. Split them. (e.g., separate "City" from "Street Address", separate "First Name" from "Full Name").
        
#         You MUST output a JSON object with a single key 'labels'. 
#         The value of 'labels' MUST be a dictionary mapping the granular category name to a rich, semantic description.
        
#         BAD Example: {{"labels": {{"location": "A place", "medical": "medical stuff", "name": "A person"}}}}
#         GOOD Example: {{"labels": {{"street_address": "A specific physical building number and street name", "city": "A general municipality or metropolitan area", "disease": "A specific medical condition, diagnosis, or illness", "api_key": "An alphanumeric string used for software authentication"}}}}
        
#         Analyze the query and generate the required granular JSON schema.
#         """
        
#         result = self._call_json(prompt).get("labels", {})
        
#         if isinstance(result, list):
#             return {item.lower(): f"Entities relating to {item}" for item in result}
#         elif isinstance(result, dict):
#             return {k.lower(): v for k, v in result.items()}
#         return {}
    
#     def _call_json(self, prompt: str) -> dict:
#         try:
#             response = self.client.chat.completions.create(
#                 messages=[
#                     {"role": "system", "content": "You are a data routing API. You must output only valid JSON."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 model=self.model_name,
#                 temperature=0.0,
#                 response_format={"type": "json_object"}
#             )
#             return json.loads(response.choices[0].message.content)
#         except Exception as e:
#             print(f"LLM Routing Error ({self.model_name}): {e}")
#             return {}

#     def process(self, state: QueryState):
#         pass


import json
from openai import OpenAI
from groq import Groq
from typing import List, Dict
from interfaces import DomainClassifier, IntentClassifier, LabelGenerator, QueryState
from config import GROQ_API_KEY, RUNPOD_OLLAMA_URL

class FastRoutingClassifier(DomainClassifier, IntentClassifier, LabelGenerator):
    def __init__(self, model_name="qwen2.5:7b"):
        self.model_name = model_name
        self.is_runpod = True

        if self.is_runpod:
            print(f"🚀 Routing Classifier: Initialized RUNPOD ({self.model_name})")
            self.client = OpenAI(
                base_url=RUNPOD_OLLAMA_URL,
                api_key="runpod_ollama"
            )
        else:
            print(f"☁️ Routing Classifier: Initialized CLOUD API ({self.model_name})")
            self.client = Groq(api_key=GROQ_API_KEY)

    def process(self, state: QueryState):
        """
        Consolidated routing logic: Domain, Intent, and Labels in a single LLM pass.
        """
        prompt = f"""
        Analyze the following user query: '{state.raw_query}'

        You are a specialized Data Routing API. Your task is to perform a three-part analysis in a single pass:

        ### 1. DOMAIN RECOGNITION
        Identify the broad industry or field (e.g., healthcare, finance, education, retail, software engineering, law, general conversation). 
        - Format: Concise, descriptive lowercase string.

        ### 2. INTENT PARSING
        Determine the primary task or user goal. 
        - Format: Verb-noun phrase in snake_case.
        - Examples: check_medication_interaction, check_account_balance, debug_code, book_flight.

        ### 3. GRANULAR LABEL GENERATION
        Identify all sensitive, confidential, or PII entity categories present in the text.
        - Universal Baseline: Always catch standard identifiers (names, emails, street addresses, government IDs).
        - Domain Specificity: Identify entities sensitive within the detected domain (e.g., medical diagnoses for healthcare, API keys for software).
        - Extreme Granularity: Split broad categories (e.g., separate "City" from "Street Address").
        - Format: A dictionary where key = category name and value = rich semantic description.

        ### OUTPUT REQUIREMENT
        Return ONLY a JSON object with these keys: "domain", "intent", "labels".
        """

        try:
            result = self._call_json(prompt)
            
            # Update state with consolidated results
            state.domain = result.get("domain", "general").lower()
            state.intent = result.get("intent", "unknown_task").lower()
            
            # Label Sanitization
            raw_labels = result.get("labels", {})
            if isinstance(raw_labels, list):
                state.potential_labels = {item.lower(): f"Entities relating to {item}" for item in raw_labels}
            elif isinstance(raw_labels, dict):
                state.potential_labels = {k.lower(): v for k, v in raw_labels.items()}
            else:
                state.potential_labels = {}

        except Exception as e:
            print(f"Combined Routing Error: {e}")
            state.domain = "general"
            state.intent = "unknown_task"
            state.potential_labels = {}

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

    # Inherited methods now call the consolidated process if needed, 
    # but the benchmark should ideally call .process(state) directly.
    def process_domain(self, state: QueryState) -> str:
        if not state.domain: self.process(state)
        return state.domain

    def process_intent(self, state: QueryState) -> str:
        if not state.intent: self.process(state)
        return state.intent

    def process_labels(self, state: QueryState) -> dict:
        if not state.potential_labels: self.process(state)
        return state.potential_labels
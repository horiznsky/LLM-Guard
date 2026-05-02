import json
from openai import OpenAI
from groq import Groq
from typing import List
from interfaces import DomainClassifier, IntentClassifier, LabelGenerator, QueryState
from config import GROQ_API_KEY, RUNPOD_OLLAMA_URL

class FastRoutingClassifier(DomainClassifier, IntentClassifier, LabelGenerator):
    # Recommend using llama3.1:8b for routing
    def __init__(self, model_name="qwen2.5:7b"):
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
        # Removed the strict hardcoded choices. Let the LLM infer the context.
        prompt = f"""
        Analyze this query: '{state.raw_query}'. 
        Determine the broad industry, field, or domain this query belongs to (e.g., healthcare, finance, education, retail, software engineering,law, general conversation).
        Output the domain as a concise, descriptive lowercase string.
        Output a JSON object with a single key 'domain'.
        """
        return self._call_json(prompt).get("domain", "general").lower()

    # In classifiers.py
    def process_intent(self, state: QueryState) -> str:
        prompt = f"""
        Query: '{state.raw_query}'
        Domain: '{state.domain}'
        
        Determine the primary intent or task of the user's query. 
        Output the intent as a concise verb-noun phrase in snake_case. 
        
        PATTERN EXAMPLES:
        Query: "What happens if I take Aspirin with my current meds?" -> {{"intent": "check_medication_interaction"}}
        Query: "How much money is in my savings?" -> {{"intent": "check_account_balance"}}
        Query: "Why is my Python script throwing a 500 error?" -> {{"intent": "debug_code"}}
        Query: "Book a flight to Paris for tomorrow." -> {{"intent": "book_flight"}}
        
        Output a JSON object with a single key 'intent' based on the query above.
        """
        return self._call_json(prompt).get("intent", "unknown_task")

    # def process_labels(self, state: QueryState) -> dict:
    #     # We explicitly demand a Key-Value pair where values are rich descriptions.
    #     # GLiNER acts on the description, not just the label name.
    #     prompt = f"""
    #     Query: '{state.raw_query}'
    #     Domain: '{state.domain}'
        
    #     Identify all sensitive or PII entity categories in this text that are contextually relevant to the Domain.
    #     You MUST output a JSON object with a single key 'labels'. 
    #     The value of 'labels' MUST be a dictionary mapping the category name to a rich, semantic description.
        
    #     BAD Example: {{"labels": {{"PERSON": "Person name"}}}}
    #     GOOD Example: {{"labels": {{"person": "The first and last name of an individual", "disease": "A specific medical condition or illness"}}}}
        
    #     Analyze the query based on its domain and generate the required JSON schema.
    #     """
        
    #     result = self._call_json(prompt).get("labels", {})
        
    #     # Safety wrapper: If the LLM still hallucinates a list, dynamically convert it
    #     if isinstance(result, list):
    #         return {item.lower(): f"Entities relating to {item}" for item in result}
    #     elif isinstance(result, dict):
    #         return {k.lower(): v for k, v in result.items()}
    #     return {}
    
    def process_labels(self, state: QueryState) -> dict:
        prompt = f"""
        Query: '{state.raw_query}'
        Domain: '{state.domain}'
        
        Identify all potential sensitive, confidential, or PII entity categories present in this text.
        You must look for BOTH Universal identifiers AND Domain-Specific sensitive data.
        
        Extraction Rules:
        1. Universal Baseline: ALWAYS identify standard direct identifiers regardless of domain (e.g., exact names, phone numbers, emails, street addresses, government IDs, financial data).
        2. Domain Specificity: ALSO identify entities that are sensitive specifically within the '{state.domain}' domain (e.g., API keys, medical diagnoses, proprietary project names).
        3. Extreme Granularity: Do not use broad categories. Split them. (e.g., separate "City" from "Street Address", separate "First Name" from "Full Name").
        
        You MUST output a JSON object with a single key 'labels'. 
        The value of 'labels' MUST be a dictionary mapping the granular category name to a rich, semantic description.
        
        BAD Example: {{"labels": {{"location": "A place", "medical": "medical stuff", "name": "A person"}}}}
        GOOD Example: {{"labels": {{"street_address": "A specific physical building number and street name", "city": "A general municipality or metropolitan area", "disease": "A specific medical condition, diagnosis, or illness", "api_key": "An alphanumeric string used for software authentication"}}}}
        
        Analyze the query and generate the required granular JSON schema.
        """
        
        result = self._call_json(prompt).get("labels", {})
        
        if isinstance(result, list):
            return {item.lower(): f"Entities relating to {item}" for item in result}
        elif isinstance(result, dict):
            return {k.lower(): v for k, v in result.items()}
        return {}
    
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
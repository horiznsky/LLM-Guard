# import json
# from typing import List
# from openai import OpenAI
# from groq import Groq
# from interfaces import PolicyEvaluator, QueryState, TokenMapping
# from config import GROQ_API_KEY, RUNPOD_OLLAMA_URL

# class LLMPolicyEvaluator(PolicyEvaluator):
#     # Recommend using qwen2.5:7b for strict policy logic
#     def __init__(self, model_name="qwen2.5:7b"):
#         self.model_name = model_name
#         self.is_runpod = True

#         if self.is_runpod:
#             print(f"🛡️ Policy Evaluator: Initialized RUNPOD ({self.model_name})")
#             self.client = OpenAI(
#                 base_url=RUNPOD_OLLAMA_URL, # Uses the proxy URL from config.py
#                 api_key="runpod_ollama"     # Dummy key, required by library
#             )
#         else:
#             print(f"☁️ Policy Evaluator: Initialized CLOUD API ({self.model_name})")
#             self.client = Groq(api_key=GROQ_API_KEY)

#     def process(self, state: QueryState) -> List[TokenMapping]:
#         if not state.extracted_entities:
#             return []

#         # Pass both token and label to the LLM so it knows exactly what the entity represents
#         entities_for_prompt = [{"token": e['token'], "label": e['label']} for e in state.extracted_entities]
#         entities_json = json.dumps(entities_for_prompt)

#         # In policy.py
#         # Update the prompt string inside the process method:
#         prompt = f"""
#         Query: "{state.raw_query}"
#         Domain: "{state.domain}"
#         Intent: "{state.intent}"
#         Entities: {entities_json}

#         You are a strict privacy policy engine.
#         Your goal: sanitize sensitive data while PRESERVING MAXIMUM UTILITY for the downstream LLM to fulfill the intent.

#         For EACH entity, assign ONE action:
#         - "abstract": Replace with its label in brackets (e.g., [PERSON], [DISEASE]). ALWAYS PREFER THIS over dropping, as it preserves context for the LLM.
#         - "drop": Use ONLY for highly sensitive direct identifiers (SSNs, exact street addresses) where even knowing the category is a risk.
#         - "keep": ONLY if the exact raw value is absolutely required to fulfill the intent (e.g., the specific city name for a weather intent, or the specific programming language for a coding intent).

#         Rules:
#         1. NEVER keep direct identifiers (names, emails, phone numbers).
#         2. Default to "abstract" if the token is sensitive but the category provides useful context.
#         3. Keep only task-critical values.

#         Output ONLY valid JSON matching this exact structure:
#         {{
#         "mappings": [
#             {{"original_token": "...", "action": "drop|abstract|keep", "replacement_token": "..."}}
#         ]
#         }}
#         """
        
#         # HIGH-PERFORMANCE, LOW-LATENCY PROMPT
# #         prompt = f"""
# # Query: "{state.raw_query}"
# # Domain: "{state.domain}"
# # Intent: "{state.intent}"
# # Entities: {entities_json}

# # You are a strict privacy policy engine.

# # Your goal: protect user privacy while preserving only the minimum information required to fulfill the intent.

# # For EACH entity, assign ONE action:

# # - "drop": Remove completely. Use for direct identifiers (names, phone numbers, IDs, exact addresses, emails).
# # - "abstract": Replace with its label in brackets (e.g., [PERSON], [LOCATION], [DATE]) if the type is useful but the exact value is not.
# # - "keep": ONLY if the exact value is absolutely required to fulfill the intent.

# # Rules:
# # 1. If unsure → choose "drop"
# # 2. NEVER keep direct identifiers
# # 3. Prefer "abstract" over "keep"
# # 4. Keep only task-critical values (e.g., disease name for diagnosis, city for weather)
# # 5. Granularity rule:
# #    - City/state → can keep if needed
# #    - Street/ZIP → drop

# # Output ONLY valid JSON matching this exact structure. 
# # For "drop", replacement_token MUST be "". 
# # For "keep", replacement_token MUST be the original_token.

# # {{
# #   "mappings": [
# #     {{"original_token": "...", "action": "drop|abstract|keep", "replacement_token": "..."}}
# #   ]
# # }}
# # """


#         try:
#             response = self.client.chat.completions.create(
#                 messages=[
#                     {"role": "system", "content": "You are a strict zero-trust privacy policy engine. Output valid JSON only."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 model=self.model_name,
#                 temperature=0.0,
#                 response_format={"type": "json_object"}
#             )

#             result = json.loads(response.choices[0].message.content)
#             mappings = []
            
#             # Match back to the original labels AND spans from GLiNER extraction
#             for mapped_item in result.get("mappings", []):
#                 # Use .get() with fallbacks in case the model forgets a key
#                 original = mapped_item.get("original_token", "")
#                 if not original:
#                     continue # Skip if the LLM hallucinated an empty item
                
#                 label = "UNKNOWN"
#                 start = -1
#                 end = -1
                
#                 # Find the corresponding entity to grab its span
#                 for e in state.extracted_entities:
#                     if e["token"] == original:
#                         label = e.get("label", "UNKNOWN")
#                         start = e.get("start", -1)
#                         end = e.get("end", -1)
#                         break
                
#                 mappings.append(TokenMapping(
#                     original_token=original,
#                     # Safely default to "" if missing
#                     replacement_token=mapped_item.get("replacement_token", ""),
#                     label=label,
#                     # Safely default to "drop" (Fail-safe) to prevent accidental data loss if it bugs out
#                     action=mapped_item.get("action", "drop").lower(), 
#                     start=start,
#                     end=end
#                 ))
#             return mappings
            
#         except Exception as e:
#             print(f"Policy Evaluation Error ({self.model_name}): {e}")
#             return []


import json
import re
from typing import List
from openai import OpenAI
from groq import Groq
from interfaces import PolicyEvaluator, QueryState, TokenMapping
from config import GROQ_API_KEY, RUNPOD_OLLAMA_URL

class LLMPolicyEvaluator(PolicyEvaluator):
    # Swapped to deepseek-r1:8b
    def __init__(self, model_name="deepseek-r1:8b"):
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
        
        # PROMPT UPDATED FOR DEEPSEEK R1
        # Notice we removed the forced "reasoning" key in the JSON, 
        # because R1 will do its reasoning in the <think> block naturally.
        # Inside policy.py (for DeepSeek-R1:8B)
        prompt = f"""
        Query: "{state.raw_query}"
        Domain: "{state.domain}"
        Intent: "{state.intent}"
        Entities: {entities_json}

        You are a Senior Privacy Architect specializing in Zero-Trust LLM Gateways.
        Your task is to sanitize the query based on the extraction list.

        THE THREE COMMANDMENTS OF CLASSIFICATION:
        For EACH entity, you MUST assign ONE of the following actions based on these strict definitions:

        1. "drop" (Zero Utility): 
        - Definition: Direct Identifiers (SSNs, phone numbers, exact addresses, full names).
        - Rule: These add NO VALUE to the query's intent and are a massive privacy risk. Drop them entirely.

        2. "abstract" (Structural Necessity): 
        - Definition: PII that is a privacy risk but is needed for sentence structure or context. 
        - Rule: Replace with its categorical label (e.g., [PERSON], [DATE], [LOCATION]). Prefer this over dropping if context is needed.

        3. "keep" (Mission Critical): 
        - Definition: Strings mechanically essential to fulfilling the Intent. 
        - Rule: If removed, the task becomes impossible (e.g., a specific city name for a weather intent, or an error code for a debugging intent). NEVER use this for direct human identifiers.

        First, think through your decisions using the Three Commandments. 
        Then, you MUST output ONLY valid JSON matching this exact structure:
        {{
        "mappings": [
            {{"original_token": "...", "action": "drop|abstract|keep", "replacement_token": "..."}}
        ]
        }}
        """
        try:
            # NOTE: Removed response_format={"type": "json_object"} because R1 needs to output <think> tags first
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.0
            )

            raw_content = response.choices[0].message.content
            
            # --- THE MAGIC PARSER FOR R1 ---
            # 1. Strip out the <think>...</think> block using Regex
            clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            
            # 2. Extract just the JSON part (in case R1 added conversational text like "Here is the JSON:")
            json_match = re.search(r'\{.*\}', clean_content, flags=re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                print("Failed to find JSON in DeepSeek output.")
                return []
            # -------------------------------

            mappings = []
            
            for mapped_item in result.get("mappings", []):
                original = mapped_item.get("original_token", "")
                if not original: continue 
                
                label = "UNKNOWN"
                start = -1
                end = -1
                
                for e in state.extracted_entities:
                    if e["token"] == original:
                        label = e.get("label", "UNKNOWN")
                        start = e.get("start", -1)
                        end = e.get("end", -1)
                        break
                
                mappings.append(TokenMapping(
                    original_token=original,
                    replacement_token=mapped_item.get("replacement_token", ""),
                    label=label,
                    action=mapped_item.get("action", "drop").lower(), 
                    start=start,
                    end=end
                ))
            return mappings
            
        except Exception as e:
            print(f"Policy Evaluation Error ({self.model_name}): {e}")
            return []
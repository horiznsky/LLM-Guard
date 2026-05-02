from google import genai
import pandas as pd
import json
import re

client = genai.Client(api_key="AIzaSyCNNVIqXkxabQ_CO5uWuFW9q7u3h2KG8nc")  # get from Google AI Studio

DOMAIN = "Finance"
COUNT = 20  # ⚠️ keep small per call

def generate_bulk_data(domain, count, max_retries=3):
    system_instruction = (
        "You are a synthetic data generator for PII benchmarking. "
        "Mimic How users ask queries in a real world scenario"
        "Return ONLY valid JSON. No markdown."
    )

    user_prompt = f"""
Generate {count} rows for '{domain}'.

Schema:
{{
    "domain": "{domain}",
    "intent": "short description",
    "original_query": "query with PII",
    "expected_S_drop": ["sensitive strings"],
    "expected_S_abstract": ["generalizable strings"],
    "expected_T_core": ["technical keywords"]
}}

Return ONLY JSON array.
"""

    full_prompt = system_instruction + "\n\n" + user_prompt

    for _ in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",  # ✅ works on free tier
                contents=full_prompt,
            )

            text = response.text.strip()
            text = re.sub(r"```json|```", "", text)

            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                text = match.group(0)

            return json.loads(text)

        except Exception as e:
            print("Retrying:", e)

    return []

if __name__ == "__main__":
    all_data = []

    for _ in range(5):  # 5 × 20 = 100 rows
        batch = generate_bulk_data(DOMAIN, 20)
        all_data.extend(batch)

    df = pd.DataFrame(all_data)
    df.to_csv("output2.csv", index=False)

    print("✅ Done (free tier)")
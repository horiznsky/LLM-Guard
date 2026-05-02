import os
import json
import time
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# === CONFIGURATION ===
INPUT_FILE = "btp_privacy_benchmark_cleaned.csv"
OUTPUT_FILE = "btp_audit_results_verified.csv"
CHECKPOINT_FILE = "audit_checkpoint.json"
API_KEY = os.environ.get('DEEPSEEK_API_KEY') # Ensure this is set in your environment
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_audit_prompts(row):
    """Constructs the Gold Standard system and user prompts."""
    system_msg = (
        "You are a Senior Privacy Architect specializing in Zero-Trust LLM Gateways. "
        "Your task is to audit a PII benchmarking dataset for technical accuracy and functional utility.\n\n"
        "THE THREE COMMANDMENTS OF CLASSIFICATION:\n"
        "1. expected_drop (Zero Utility): Direct Identifiers like SSNs, phone numbers, and full names that add NO VALUE to the query's intent.\n"
        "2. expected_abstract (Structural Necessity): PII that is a privacy risk but needed for sentence structure. Replace with [PERSON], [DATE], or [LOCATION].\n"
        "3. tcore (Mission Critical): Strings mechanically essential to answering the query. If removed, the task becomes impossible."
    )
    
    user_msg = f"""
    AUDIT REQUEST:
    Input Data:
    - User Query: "{row['original_query']}"
    - Current Drop: {row['expected_S_drop']}
    - Current Abstract: {row['expected_S_abstract']}
    - Current Core: {row['expected_T_core']}

    INSTRUCTIONS:
    1. Scan for Shadow PII: Identify PII strings NOT present in any current list.
    2. Validate Logic: Ensure classification follows the Three Commandments.
    3. Inference Check: Look for unique job titles or context combinations that reveal identity.

    OUTPUT FORMAT (STRICT JSON):
    {{
      "audit_status": "PASS" | "FAIL",
      "missed_entities": [],
      "misclassified_entities": [{{ "entity": "string", "from": "current_cat", "to": "new_cat", "reason": "why" }}],
      "final_verified_lists": {{
        "drop": [],
        "abstract": [],
        "tcore": []
      }},
      "confidence_score": 0.0-1.0
    }}
    """
    return system_msg, user_msg

def audit_row(row):
    """Calls DeepSeek-V4-Pro with thinking/reasoning enabled."""
    system_msg, user_msg = get_audit_prompts(row)
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
            response_format={'type': 'json_object'}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error auditing row: {e}")
        return None

def main():
    # 1. Load and Clean Data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    df = df.drop_duplicates(subset=['original_query']).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 2. Handle Checkpoints
    start_index = 0
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            start_index = json.load(f).get('last_processed_index', 0) + 1
            print(f"Resuming from index {start_index}...")

    results = []
    
    # 3. Processing Loop
    print(f"🚀 Starting DeepSeek-V4-Pro Audit for {len(df) - start_index} rows...")
    
    try:
        for i in tqdm(range(start_index, len(df)), desc="Auditing PII", unit="row"):
            row = df.iloc[i]
            audit_result = audit_row(row)
            
            if audit_result:
                audit_result['original_query'] = row['original_query']
                audit_result['intent'] = row['intent']
                audit_result['domain'] = row['domain']
                results.append(audit_result)
            
            # Save progress every 10 rows
            if i % 10 == 0:
                pd.DataFrame(results).to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE), index=False)
                results = [] # Clear memory
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump({'last_processed_index': i}, f)
                print(f"✅ Processed {i}/{len(df)} rows. Checkpoint saved.")

            # Respect API limits
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping script... progress saved.")
    
    print(f"✨ Audit complete. Final results: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
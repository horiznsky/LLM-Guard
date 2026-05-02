import pandas as pd
import ast
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# Import your pipeline components from the files we created earlier
from interfaces import QueryState
from classifiers import FastRoutingClassifier
from extractors import GlinerExtractor
from policy import LLMPolicyEvaluator

print("Loading Semantic Evaluation Model...")
semantic_model = SentenceTransformer('all-mpnet-base-v2')

# In benchmark.py
def is_semantically_correct(expected: str, predicted: str, threshold=0.75) -> bool:
    if not expected or not predicted:
        return False
        
    # Clean snake_case into normal words for better semantic embeddings
    exp_clean = expected.replace("_", " ")
    pred_clean = predicted.replace("_", " ")
    
    if exp_clean in pred_clean or pred_clean in exp_clean:
        return True
        
    emb_exp = semantic_model.encode(exp_clean, convert_to_tensor=True)
    emb_pred = semantic_model.encode(pred_clean, convert_to_tensor=True)
    
    similarity = util.cos_sim(emb_exp, emb_pred).item()
    return similarity >= threshold

def safe_eval(val):
    """Safely convert string representation of lists from CSV into actual Python lists."""
    if pd.isna(val):
        return []
    try:
        # Some rows might have malformed strings, clean them up
        val = str(val).replace('\r', '').replace('\n', '')
        parsed = ast.literal_eval(val)
        return [str(x).strip().lower() for x in parsed]
    except:
        return []

def calculate_metrics(tp, fp, fn):
    """Calculates standard NLP and Privacy metrics based on the B.Tech Final PDF."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Overmasking: Proportion of our masked tokens that were actually safe (1 - Precision)
    overmasking_rate = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Undermasking: Proportion of actual sensitive tokens that we missed (1 - Recall)
    undermasking_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall, f1, overmasking_rate, undermasking_rate

def run_benchmark(csv_path: str, num_samples: int = 100):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Limit samples for testing if needed
    if num_samples and num_samples < len(df):
        df = df.sample(num_samples, random_state=42).copy()

    # Initialize Pipeline Components
    print("Initializing Models (GPU/API)...")
    router = FastRoutingClassifier()
    extractor = GlinerExtractor()
    policy = LLMPolicyEvaluator()

    # Tracking Counters
    results = {
        "domain_correct": 0,
        "intent_correct": 0,
        "total_queries": len(df),
        "tp": 0, "fp": 0, "fn": 0
    }

    print("\nStarting Benchmark Loop...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        raw_query = str(row['original_query'])
        expected_domain = str(row['domain']).strip().lower()
        expected_intent = str(row['intent']).strip().lower()
        
        # Ground Truth Privacy Sets (combine drop and abstract as 'Sensitive')
        expected_s_drop = safe_eval(row['expected_S_drop'])
        expected_s_abs = safe_eval(row['expected_S_abstract'])
        ground_truth_sensitive = set(expected_s_drop + expected_s_abs)

        # --- 1. Run Pipeline ---
        state = QueryState(query_id=str(idx), raw_query=raw_query)
        
        try:
            state.domain = router.process_domain(state)
            state.intent = router.process_intent(state)
            state.potential_labels = router.process_labels(state)
            state.extracted_entities = extractor.process(state)
            state.mappings = policy.process(state)
            
            # Optional: Save state to JSONL for debugging individual failures
            # from state_logger import save_query_state
            # save_query_state(state, phase="benchmark_eval")
            
        except Exception as e:
            print(f"Error processing query {idx}: {e}")
            continue

        # --- 2. Evaluate Domain & Intent ---
        pred_domain = str(state.domain).strip().lower()
        pred_intent = str(state.intent).strip().lower()
        
        if is_semantically_correct(expected_domain, pred_domain, threshold=0.80):
            results["domain_correct"] += 1
            
        # Intent matching can be fuzzy since LLMs might rephrase slightly
        if is_semantically_correct(expected_intent, pred_intent, threshold=0.65):
            results["intent_correct"] += 1
        else:
            # ADD THIS DEBUG PRINT
            print(f"\n[INTENT FAILED] Expected: '{expected_intent}' | Qwen Guessed: '{pred_intent}'")

        # --- 3. Evaluate Privacy Tokens (Overlap Matching instead of Strict) ---
        predicted_sensitive = []
        for mapping in state.mappings:
            if mapping.action in ['drop', 'abstract', 'fake']:
                predicted_sensitive.append(mapping.original_token.strip().lower())

        query_tp = 0
        query_fp = 0
        
        # Create a copy to track what we've matched
        unmatched_ground_truth = list(ground_truth_sensitive)

        # Calculate True Positives (with partial overlap) and False Positives
        for pred in predicted_sensitive:
            matched = False
            for gt in unmatched_ground_truth:
                # If the prediction is inside the ground truth OR ground truth inside prediction
                if pred in gt or gt in pred:
                    query_tp += 1
                    unmatched_ground_truth.remove(gt) # Remove to avoid double counting
                    matched = True
                    break
            
            if not matched:
                query_fp += 1 # We predicted it, but it wasn't in ground truth (Overmasking)

        # Whatever is left in ground truth was missed (False Negatives)
        query_fn = len(unmatched_ground_truth)

        results["tp"] += query_tp
        results["fp"] += query_fp
        results["fn"] += query_fn
        
        # Respect API rate limits (Groq/Gemini have limits per minute on free tiers)
        time.sleep(1)

    # --- 4. Final Math & Report ---
    domain_acc = results["domain_correct"] / results["total_queries"]
    intent_acc = results["intent_correct"] / results["total_queries"]
    
    precision, recall, f1, overmasking, undermasking = calculate_metrics(
        results["tp"], results["fp"], results["fn"]
    )

    print("\n" + "="*50)
    print(" 🚀 BENCHMARK RESULTS (PRE-GENERATION) 🚀")
    print("="*50)
    print(f"Total Queries Evaluated : {results['total_queries']}")
    print(f"Domain Accuracy         : {domain_acc:.2%}")
    print(f"Intent Accuracy         : {intent_acc:.2%}")
    print("-" * 50)
    print(" 🛡️ PRIVACY METRICS (vs Ideal Range in PDF)")
    print("-" * 50)
    print(f"Precision               : {precision:.3f}  (Ideal: ~0.90)")
    print(f"Recall                  : {recall:.3f}  (Ideal: >= 0.90)")
    print(f"F1 Score                : {f1:.3f}  (Ideal: >= 0.90)")
    print(f"Overmasking Rate (FP)   : {overmasking:.3f}  (Ideal: <= 0.10)")
    print(f"Undermasking Rate (FN)  : {undermasking:.3f}  (Ideal: <= 0.10)")
    print("="*50)

if __name__ == "__main__":
    DATASET_PATH = "btp_privacy_benchmark_final.csv"
    
    run_benchmark(DATASET_PATH, num_samples=30)
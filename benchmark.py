import pandas as pd
import ast
import json
import re
from datetime import datetime
from dataclasses import asdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, util

# Import pipeline components
from interfaces import QueryState
from classifiers import FastRoutingClassifier
from extractors import GlinerExtractor
from policy import LLMPolicyEvaluator
from llm_clients import RunPodLLM  # Updated to use RunPod for Qwen

# ==========================================
# 1. UTILITIES & SEMANTIC MODELS
# ==========================================

print("Loading MPNet Semantic Model for Metrics...")
# semantic_model = SentenceTransformer('all-mpnet-base-v2')
semantic_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

def perfect_sanitize(raw_query: str, mappings: list) -> str:
    """
    Sanitizes a string using precise reverse-index slicing to prevent substring collision.
    """
    # Only process tokens that are actively changing the string
    active_mappings = [m for m in mappings if m.action in ['drop', 'abstract', 'fake']]
    
    if not active_mappings:
        return raw_query

    # Separate mappings with valid coordinates vs invalid ones (fallback)
    valid_coords = [m for m in active_mappings if m.start != -1 and m.end != -1]
    missing_coords = [m for m in active_mappings if m.start == -1 or m.end == -1]

    sanitized = raw_query

    # Phase 1: Reverse-Index Slicing (Prevents index drift)
    valid_coords.sort(key=lambda x: x.start, reverse=True)
    for m in valid_coords:
        if sanitized[m.start:m.end] == m.original_token:
            sanitized = sanitized[:m.start] + m.replacement_token + sanitized[m.end:]
        else:
            missing_coords.append(m)

    # Phase 2: Fallback (Length-Sorted .replace)
    if missing_coords:
        missing_coords.sort(key=lambda x: len(x.original_token), reverse=True)
        for m in missing_coords:
            sanitized = sanitized.replace(m.original_token, m.replacement_token)

    # Phase 3: Cleanup consecutive spaces
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized

def save_query_state(state: QueryState, filepath: str = "query_states_log.jsonl", phase: str = "final"):
    try:
        state_dict = asdict(state)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "state": state_dict
        }
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"Failed to save state for Query ID {state.query_id}: {e}")

def safe_eval(val):
    if pd.isna(val): return []
    try:
        val = str(val).replace('\r', '').replace('\n', '')
        return [str(x).strip().lower() for x in ast.literal_eval(val)]
    except:
        return []

def is_semantically_correct(expected: str, predicted: str, threshold=0.50) -> bool:
    if not expected or not predicted: return False
    exp_clean, pred_clean = expected.replace("_", " "), predicted.replace("_", " ")
    if exp_clean in pred_clean or pred_clean in exp_clean: return True
    
    emb_exp = semantic_model.encode(exp_clean, convert_to_tensor=True)
    emb_pred = semantic_model.encode(pred_clean, convert_to_tensor=True)
    return util.cos_sim(emb_exp, emb_pred).item() >= threshold

# ==========================================
# 2. THE PROCESSING WORKER
# ==========================================

def process_single_query(idx, row, router, extractor, policy, generator_llm):
    """Executes the full gateway and generation pipeline."""
    raw_query = str(row['original_query'])
    state = QueryState(query_id=str(idx), raw_query=raw_query)
    
    expected_s_drop = safe_eval(row['expected_S_drop'])
    expected_s_abs = safe_eval(row['expected_S_abstract'])
    ground_truth_sensitive = set(expected_s_drop + expected_s_abs)
    
    metrics = {
        "idx": idx, "query_tp": 0, "query_fp": 0, "query_fn": 0,
        "domain_correct": 0, "intent_correct": 0,
        "ips_score": 0.0, "aes_score": 0.0, "semantic_score": 0.0, "output_leakage": 0
    }

    try:
        # --- A. GATEWAY PIPELINE ---
        # state.domain = router.process_domain(state)
        # state.intent = router.process_intent(state)
        # state.potential_labels = router.process_labels(state)
        router.process(state)
        state.extracted_entities = extractor.process(state)
        state.mappings = policy.process(state)
        
        # Build Sanitized Query using perfect_sanitize logic
        state.sanitized_query = perfect_sanitize(state.raw_query, state.mappings)

        # --- B. GENERATION PIPELINE ---
        # 1. Generate Baseline Answer (RAW)
        state.llm_raw_response = generator_llm.generate(state.raw_query)
        
        # 2. Generate Privacy Answer (SANITIZED)
        state.llm_sanitized_response = generator_llm.generate(state.sanitized_query)
        
        # 3. Restore the sanitized response
        final_text = state.llm_sanitized_response
        # Sort mappings by length to prevent substring collision during restoration
        sorted_mappings = sorted(state.mappings, key=lambda m: len(m.original_token), reverse=True)
        for mapping in sorted_mappings:
            if mapping.action in ['abstract', 'fake']:
                rep_token = mapping.replacement_token if mapping.replacement_token else f"[{mapping.label.upper()}]"
                if rep_token:
                    final_text = final_text.replace(rep_token, mapping.original_token)
        state.final_restored_response = final_text

        # --- C. METRICS CALCULATION ---
        # 1. Routing Metrics
        if is_semantically_correct(str(row['domain']).lower(), str(state.domain).lower(), 0.80):
            metrics["domain_correct"] = 1
        if is_semantically_correct(str(row['intent']).lower(), str(state.intent).lower(), 0.50):
            metrics["intent_correct"] = 1

        # 2. Privacy Token Metrics
        predicted_sensitive = [m.original_token.strip().lower() for m in state.mappings if m.action in ['drop', 'abstract', 'fake']]
        unmatched_gt = list(ground_truth_sensitive)

        for pred in predicted_sensitive:
            matched = False
            for gt in unmatched_gt:
                if pred in gt or gt in pred:
                    metrics["query_tp"] += 1
                    unmatched_gt.remove(gt)
                    matched = True
                    break
            if not matched: metrics["query_fp"] += 1
        
        metrics["query_fn"] = len(unmatched_gt)
        
        # 3. Advanced Embeddings Metrics
        emb_raw_q = semantic_model.encode(state.raw_query, convert_to_tensor=True)
        emb_san_q = semantic_model.encode(state.sanitized_query, convert_to_tensor=True)
        emb_raw_resp = semantic_model.encode(state.llm_raw_response, convert_to_tensor=True)
        emb_restored = semantic_model.encode(state.final_restored_response, convert_to_tensor=True)

        metrics["ips_score"] = util.cos_sim(emb_raw_q, emb_san_q).item()
        metrics["aes_score"] = util.cos_sim(emb_raw_resp, emb_restored).item()
        metrics["semantic_score"] = util.cos_sim(emb_raw_q, emb_restored).item()

        # 4. Output Leakage Check
        llm_sanitized_lower = state.llm_sanitized_response.lower()
        for gt in ground_truth_sensitive:
            if gt in llm_sanitized_lower:
                metrics["output_leakage"] = 1
                break

        save_query_state(state, phase="completed")
        return metrics

    except Exception as e:
        print(f"Error processing query {idx}: {e}")
        return None

# ==========================================
# 3. MAIN ORCHESTRATOR
# ==========================================

def run_benchmark(csv_path: str, num_samples: int = 30):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    if num_samples and num_samples < len(df):
        df = df.sample(num_samples, random_state=42).copy()

    print("Initializing Gateway Components...")
    router = FastRoutingClassifier()
    extractor = GlinerExtractor()
    policy = LLMPolicyEvaluator()
    
    # UPDATED: Using Qwen 2.5 72B loaded on RunPod
    generator_llm = RunPodLLM(model_name="qwen2.5:72b") 

    agg = {
        "total": 0, "domain": 0, "intent": 0, "tp": 0, "fp": 0, "fn": 0,
        "ips_sum": 0.0, "aes_sum": 0.0, "semantic_sum": 0.0, "leakage_count": 0
    }

    print(f"\nStarting Benchmark with Qwen2.5-72B (RunPod)...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single_query, idx, row, router, extractor, policy, generator_llm): idx for idx, row in df.iterrows()}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            if res:
                agg["total"] += 1
                agg["domain"] += res["domain_correct"]
                agg["intent"] += res["intent_correct"]
                agg["tp"] += res["query_tp"]
                agg["fp"] += res["query_fp"]
                agg["fn"] += res["query_fn"]
                agg["ips_sum"] += res["ips_score"]
                agg["aes_sum"] += res["aes_score"]
                agg["semantic_sum"] += res["semantic_score"]
                agg["leakage_count"] += res["output_leakage"]

    # --- FINAL MATH ---
    if agg["total"] > 0:
        precision = agg["tp"] / (agg["tp"] + agg["fp"]) if (agg["tp"] + agg["fp"]) > 0 else 0.0
        recall = agg["tp"] / (agg["tp"] + agg["fn"]) if (agg["tp"] + agg["fn"]) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print("\n" + "="*60)
        print(" 🚀 BENCHMARK SUMMARY (Qwen2.5-72B @ RunPod) 🚀")
        print("="*60)
        print(f"Total Queries        : {agg['total']}")
        print(f"Domain Accuracy      : {agg['domain'] / agg['total']:.2%}")
        print(f"Intent Accuracy      : {agg['intent'] / agg['total']:.2%}")
        print("-" * 60)
        print(f"Precision            : {precision:.3f}")
        print(f"Recall               : {recall:.3f}")
        print(f"F1 Score             : {f1:.3f}")
        print("-" * 60)
        print(f"IPS (Intent Pres.)   : {agg['ips_sum'] / agg['total']:.3f}")
        print(f"AES (Answer Equiv.)  : {agg['aes_sum'] / agg['total']:.3f}")
        print(f"Leakage Rate         : {agg['leakage_count'] / agg['total']:.2%}")
        print("="*60)

if __name__ == "__main__":
    DATASET_PATH = "btp_privacy_benchmark_final.csv"
    run_benchmark(DATASET_PATH, num_samples=500)
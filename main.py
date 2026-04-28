from interfaces import QueryState
from llm_clients import GroqLLM
from classifiers import FastRoutingClassifier
from extractors import GlinerExtractor
from policy import LLMPolicyEvaluator

def run_pipeline():
    print("Initializing models (GLiNER might download on first run)...")
    router = FastRoutingClassifier()
    extractor = GlinerExtractor()
    policy = LLMPolicyEvaluator()
    final_llm = GroqLLM()

    # The test query
    raw_query = "Tell me the weather in Gorakhpur, and note that my blood type is O-positive."
    print(f"\n[RAW QUERY] {raw_query}")
    
    state = QueryState(query_id="test_001", raw_query=raw_query)

    print("\n--- Running Pipeline ---")
    
    state.domain = router.process_domain(state)
    print(f"Domain: {state.domain}")
    
    state.intent = router.process_intent(state)
    print(f"Intent: {state.intent}")
    
    state.potential_labels = router.process_labels(state)
    print(f"Potential Labels: {state.potential_labels}")

    state.extracted_entities = extractor.process(state)
    print(f"Extracted Entities: {state.extracted_entities}")

    state.mappings = policy.process(state)
    print(f"Policy Mappings:")
    for m in state.mappings:
        print(f"  - {m.original_token} -> {m.action} ({m.replacement_token})")

    # The Sanitization Step
    sanitized_text = state.raw_query
    for mapping in state.mappings:
        if mapping.action != 'keep':
            sanitized_text = sanitized_text.replace(
                mapping.original_token, 
                mapping.replacement_token
            )
    state.sanitized_query = sanitized_text
    print(f"\n[SANITIZED QUERY FOR LLM] {state.sanitized_query}")

    # Send to the Target LLM
    print("\nGenerating LLM Response (This goes to the API)...")
    state.llm_sanitized_response = final_llm.generate(state.sanitized_query)
    print(f"\n[RAW LLM RESPONSE] \n{state.llm_sanitized_response}")

    # The Restoration Step (De-anonymization)
    final_text = state.llm_sanitized_response
    for mapping in state.mappings:
        if mapping.action == 'abstract' or mapping.action == 'fake':
            final_text = final_text.replace(
                mapping.replacement_token, 
                mapping.original_token
            )
    
    state.final_restored_response = final_text
    print(f"\n[FINAL RESTORED RESPONSE] \n{state.final_restored_response}")

if __name__ == "__main__":
    run_pipeline()
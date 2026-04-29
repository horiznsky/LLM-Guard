import time
from llm_clients import RunPodLLM
from classifiers import FastRoutingClassifier

# The exact names of all the models you pulled on RunPod
models_to_test = [
    "llama3.1:8b",
    "qwen2.5:7b",
    "mistral-nemo",
    "gemma2:9b",
    "llama3.3:70b",
    "qwen2.5:72b",
    "mixtral"
]

print("==========================================")
print(" 🚀 RUNPOD OLLAMA DIAGNOSTIC SUITE 🚀 ")
print("==========================================")

for model_name in models_to_test:
    print(f"\nLoading: [{model_name}]...")
    
    # ---------------------------------------------------------
    # TEST 1: Standard Generation (Tests raw compute/loading)
    # ---------------------------------------------------------
    print(f"  -> Testing Generation... ", end="", flush=True)
    try:
        t0 = time.time()
        final_llm = RunPodLLM(model_name=model_name)
        response = final_llm.generate("Respond with exactly one word: 'Ready'.")
        t1 = time.time()
        print(f"✅ Success ({t1-t0:.2f}s) | Output: {response.strip()}")
    except Exception as e:
        print(f"❌ Failed | Error: {e}")

    # ---------------------------------------------------------
    # TEST 2: JSON Constraint (Tests routing/policy capability)
    # ---------------------------------------------------------
    print(f"  -> Testing JSON Mode...  ", end="", flush=True)
    try:
        t0 = time.time()
        router = FastRoutingClassifier(model_name=model_name)
        
        # Suppress the initialization print statement just to keep logs clean
        prompt = "Output valid JSON with a single key 'status' and value 'ok'."
        json_response = router._call_json(prompt)
        
        t1 = time.time()
        print(f"✅ Success ({t1-t0:.2f}s) | Output: {json_response}")
    except Exception as e:
        print(f"❌ Failed | Error: {e}")

print("\n==========================================")
print(" 🏁 ALL TESTS COMPLETE 🏁 ")
print("==========================================")
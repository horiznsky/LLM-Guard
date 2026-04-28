import json
import os
from datetime import datetime
from dataclasses import asdict
from interfaces import QueryState

def save_query_state(state: QueryState, filepath: str = "query_states_log.jsonl", phase: str = "final"):
    """
    Saves a QueryState object to a JSON Lines file.
    
    Args:
        state (QueryState): The state object to save.
        filepath (str): The path to the log file. Defaults to 'query_states_log.jsonl'.
        phase (str): An optional tag to know WHEN in the pipeline this was saved (e.g., 'post_ner', 'final').
    """
    try:
        # Convert the dataclass (and nested TokenMappings) to a native dictionary
        state_dict = asdict(state)
        
        # Wrap it with some useful metadata
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "state": state_dict
        }
        
        # Open in 'append' mode ('a') so we don't overwrite previous logs
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    except Exception as e:
        print(f"Failed to save state for Query ID {state.query_id}: {e}")

def load_query_states(filepath: str = "query_states_log.jsonl"):
    """
    Utility function to read the saved states back into Python for analysis.
    Returns a list of dictionaries.
    """
    history = []
    if not os.path.exists(filepath):
        print(f"No log file found at {filepath}")
        return history
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                history.append(json.loads(line.strip()))
                
    return history
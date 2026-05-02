import pandas as pd
import ast

def finalize_dataset(input_csv, output_csv):
    # Load the audited results
    df = pd.read_csv(input_csv)
    
    # Helper to parse the stringified dictionary from the audit
    def extract_list(val, key):
        try:
            # ast.literal_eval safely handles the Python-style dict strings
            data = ast.literal_eval(val)
            return data.get(key, [])
        except:
            return []

    # Map the audited lists back to your benchmark columns
    df['expected_S_drop'] = df['final_verified_lists'].apply(lambda x: extract_list(x, 'drop'))
    df['expected_S_abstract'] = df['final_verified_lists'].apply(lambda x: extract_list(x, 'abstract'))
    df['expected_T_core'] = df['final_verified_lists'].apply(lambda x: extract_list(x, 'tcore'))

    # Select only the target columns in the required order
    target_columns = [
        'domain', 
        'intent', 
        'original_query', 
        'expected_S_drop', 
        'expected_S_abstract', 
        'expected_T_core'
    ]
    
    df_final = df[target_columns]
    
    # Save the polished ground-truth dataset
    df_final.to_csv(output_csv, index=False)
    print(f"✨ Finalized dataset saved to {output_csv}")

if __name__ == "__main__":
    finalize_dataset('btp_audit_results_verified.csv', 'btp_privacy_benchmark_final.csv')
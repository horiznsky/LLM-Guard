import pandas as pd

# Define file names
input_file = "btp_privacy_benchmark_1000 - btp_privacy_benchmark_1000.csv.csv"
output_file = "btp_privacy_benchmark_cleaned.csv"

def process_benchmark_data():
    try:
        # 1. Load the dataset
        df = pd.read_csv(input_file)
        initial_count = len(df)
        
        # 2. Remove duplicates based on the 'original_query' column
        df_cleaned = df.drop_duplicates(subset=['original_query'], keep='first')
        after_dedup_count = len(df_cleaned)
        
        # 3. Randomize the rows
        # frac=1 shuffles 100% of the data; reset_index(drop=True) cleans up the row numbers
        df_final = df_cleaned.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 4. Save the cleaned and randomized version
        df_final.to_csv(output_file, index=False)
        
        # 5. Output Results
        print(f"--- Processing Complete ---")
        print(f"Initial rows:        {initial_count}")
        print(f"Duplicates removed:  {initial_count - after_dedup_count}")
        print(f"Final dataset size:  {after_dedup_count}")
        print(f"\n--- Rows Per Domain ---")
        print(df_final['domain'].value_counts())
        print(f"\nCleaned and randomized file saved as: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    process_benchmark_data()
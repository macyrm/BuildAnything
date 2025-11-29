import pandas as pd
import os
import random
from transformers import AutoTokenizer
from datasets import Dataset

OUTPUT_DIR="./processed_data"
BASE_MODEL="distilbert-base-uncased"
COMBINED_DATA_PATH = "./assets/alzheimers_data_smote_update.csv"

# The template used for generating the text feature
TEXT_TEMPLATE = (
    "Patient is a {Age}-year-old with an MMSE score of {MMSE}. "
    "Reported symptoms include memory complaints: {MemoryComplaints}, "
    "confusion: {Confusion}, disorientation: {Disorientation}, "
    "and difficulty performing daily tasks: {DifficultyCompletingTasks}."
)

def load_combined_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Error: The combined data file was not found at {data_path}. "
            "Please ensure the file is in your 'assets' folder."
        )
        
    print(f"Loading combined data from {data_path}...")
    df_raw = pd.read_csv(data_path)
    
    if 'text' not in df_raw.columns:
        print("Text column missing. Re-synthesizing 'text' feature...")
        
        required_cols = ['Age', 'MMSE', 'MemoryComplaints', 'Confusion', 'Disorientation', 'DifficultyCompletingTasks']
        if not all(col in df_raw.columns for col in required_cols):
             raise KeyError(
                f"Cannot re-synthesize 'text'. The data file is missing required columns. "
                f"Missing one of: {', '.join(c for c in required_cols if c not in df_raw.columns)}"
            )

        df_raw['text'] = df_raw.apply(
            lambda row: TEXT_TEMPLATE.format(**row), axis=1
        )
    
    print("Diagnosis Value Counts in Combined Data (0: Low Risk, 1: High Risk):")
    print(df_raw['Diagnosis'].value_counts()) 
    
    return df_raw[['text', 'Diagnosis']]

def create_datasets(df_raw):
    dataset = Dataset.from_pandas(df_raw)
    
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def tokenize_function(train_dataset, test_dataset):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_test = test_dataset.map(tokenize, batched=True)
    
    tokenized_train = tokenized_train.remove_columns(["text"])
    tokenized_test = tokenized_test.remove_columns(["text"])

    tokenized_train = tokenized_train.rename_column("Diagnosis", "labels")
    tokenized_test = tokenized_test.rename_column("Diagnosis", "labels")

    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")
    
    return tokenized_train, tokenized_test

def main():
    df_raw = load_combined_data(COMBINED_DATA_PATH)
    
    train_dataset, test_dataset = create_datasets(df_raw)
    
    MAX_SAMPLES = 300
    if len(train_dataset) > MAX_SAMPLES:
        print(f"Sampling down training data from {len(train_dataset)} to {MAX_SAMPLES} records for ultra-fast training.")
        train_dataset = train_dataset.select(range(MAX_SAMPLES))

    tokenized_train, tokenized_test = tokenize_function(train_dataset, test_dataset)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    tokenized_train.save_to_disk(os.path.join(OUTPUT_DIR, "train_dataset"))
    tokenized_test.save_to_disk(os.path.join(OUTPUT_DIR, "test_dataset"))
    print(f"\nTokenized datasets saved to {OUTPUT_DIR}/train_dataset and {OUTPUT_DIR}/test_dataset")

if __name__ == "__main__":
    main()
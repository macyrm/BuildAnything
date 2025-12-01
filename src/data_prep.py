import pandas as pd
import os
from sklearn.model_selection import train_test_split

DATA_PATH="./assets/alzheimers_disease_data_feat_engineer.csv"
TARGET_COLUMN="Diagnosis"
RANDOM_STATE=42

FEATURE_COLUMNS=[
    'Age',
    'MMSE',
    'Obesity',
    'MedicalHistory',
    'CognitiveFunction',
    'Symptoms'
]

def load_and_prepare_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Error: The data file was not found at {DATA_PATH}. "
            "Please ensure the file is in the current directory."
        )
    print(f"Loading data from {DATA_PATH}...")
    df_raw = pd.read_csv(DATA_PATH)
    X = df_raw[FEATURE_COLUMNS]
    y = df_raw[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Data prepared using {len(FEATURE_COLUMNS)} features.")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def main():
    load_and_prepare_data()

if __name__ == "__main__":
    main()
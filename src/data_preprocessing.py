import pandas as pd
import os
import json
from config import DATA_RAW_PATH, DATA_PROCESSED_PATH

def load_faq_dataset(filename="Mental_Health_FAQ.csv"):
    """Load the FAQ dataset (CSV) from the raw data folder."""
    path = os.path.join(DATA_RAW_PATH, filename)
    return pd.read_csv(path)

def load_mental_health_dataset(filename="Combined_Data.csv"):
    """Load the additional mental health dataset (CSV) from the raw data folder."""
    path = os.path.join(DATA_RAW_PATH, filename)
    return pd.read_csv(path)

def load_kb_dataset(filename="KB.json"):
    """Load the knowledge base from the raw data folder (JSON)."""
    path = os.path.join(DATA_RAW_PATH, filename)
    with open(path, "r", encoding="utf-8") as f:
        kb = json.load(f)
    return kb

def preprocess_faq(df):
    """
    Preprocess the FAQ dataset.
    Assumes it has columns: 'Question_ID', 'Questions', 'Answers'
    """
    df = df.dropna(subset=['Questions', 'Answers'])
    df['input_text'] = df['Questions']
    df['target_text'] = df['Answers']
    return df[['input_text', 'target_text']]

def preprocess_mental_health(df):
    """
    Preprocess the Mental Health dataset.
    Previously assumed it had a column 'text', but the file actually has 'statement'.
    This function now creates pseudo dialogue pairs from 'statement'.
    """
    # Drop rows where 'statement' is NaN
    df = df.dropna(subset=['statement'])

    # Create pseudo dialogue pairs
    df['input_text'] = df['statement']
    df['target_text'] = df['statement']  # For demonstration, we use the same text as target

    return df[['input_text', 'target_text']]

def merge_dialogue_datasets(faq_df, mh_df):
    """Merge the FAQ and Mental Health dialogue datasets."""
    merged_df = pd.concat([faq_df, mh_df], ignore_index=True)
    return merged_df

def save_merged_dataset(df, filename="merged_dialogue_dataset.csv"):
    """Save the merged dialogue dataset to the processed data folder."""
    output_path = os.path.join(DATA_PROCESSED_PATH, filename)
    df.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    # Load datasets
    faq_df = load_faq_dataset()
    mh_df = load_mental_health_dataset()

    # Preprocess datasets
    faq_processed = preprocess_faq(faq_df)
    mh_processed = preprocess_mental_health(mh_df)

    # Merge and save
    merged_df = merge_dialogue_datasets(faq_processed, mh_processed)
    output_file = save_merged_dataset(merged_df)
    print("Merged dialogue dataset saved to:", output_file)

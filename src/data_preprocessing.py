import pandas as pd
from src.config import DATA_RAW_PATH, DATA_PROCESSED_PATH

def load_raw_data():
    """Load the raw Crisis Text Line dataset."""
    return pd.read_csv(DATA_RAW_PATH)

def preprocess_data(df):
    """
    Preprocess the dataset:
      - Remove any sensitive information.
      - Clean and format the text.
      - Split the data into input (context) and target (response) for dialogue training.
    """
    # Example: simply drop rows with missing values and select relevant columns.
    df = df.dropna()
    # Assume the dataset has columns 'text' and 'label' for this example.
    # You may need to adapt this based on the actual dataset structure.
    df['input_text'] = df['text']  # For fine-tuning, you might create input/response pairs.
    df['target_text'] = df['text']  # This is just an example; in practice, you would have conversational pairs.
    return df[['input_text', 'target_text']]

def save_processed_data(df):
    """Save the processed dataset for fine-tuning."""
    df.to_csv(DATA_PROCESSED_PATH, index=False)

if __name__ == "__main__":
    raw_df = load_raw_data()
    processed_df = preprocess_data(raw_df)
    save_processed_data(processed_df)
    print("Data preprocessing complete. Processed data saved to:", DATA_PROCESSED_PATH)

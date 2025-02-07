from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from config import SENTIMENT_MODEL_NAME, SENTIMENT_MODEL_OUTPUT_DIR
import os

def tokenize_sentiment(examples, tokenizer):
    # Assume sentiment dataset has columns 'text' and 'status' (or a similar label column)
    return tokenizer(examples["statement"], padding="max_length", truncation=True, max_length=128)

def fine_tune_sentiment_model():
    # Load sentiment dataset from raw or processed folder
    # For demonstration, assume the dataset is in CSV format in data/raw/sentiment_dataset.csv
    dataset = load_dataset('csv', data_files={'train': os.path.join("../data/raw", "Mental_Health_FAQ.csv")}, split='train')
    
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    tokenized_dataset = dataset.map(lambda x: tokenize_sentiment(x, tokenizer), batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME, num_labels=3)  # Example: Negative, Neutral, Positive
    
    training_args = TrainingArguments(
        output_dir=SENTIMENT_MODEL_OUTPUT_DIR,
        evaluation_strategy="no",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=500,
        logging_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    trainer.save_model(SENTIMENT_MODEL_OUTPUT_DIR)
    print("Sentiment model fine-tuning complete. Model saved to:", SENTIMENT_MODEL_OUTPUT_DIR)

if __name__ == "__main__":
    fine_tune_sentiment_model()
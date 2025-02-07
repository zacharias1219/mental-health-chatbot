import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from config import DIALOGUE_MODEL_NAME, MODEL_OUTPUT_DIR

def tokenize_function(examples, tokenizer):
    inputs = examples["input_text"]
    targets = examples["target_text"]
    # Tokenize inputs (question/user text)
    model_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=128
    )
    # Tokenize targets (answer/response text)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            padding="max_length",
            truncation=True,
            max_length=128
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def fine_tune_dialogue_model():
    # 1. Load the merged CSV into a pandas DataFrame
    #    Adjust the path if your directory structure is different
    merged_csv_path = os.path.join("..", "data", "processed", "merged_dialogue_dataset.csv")
    df = pd.read_csv(merged_csv_path, encoding="utf-8")

    # 2. Convert the DataFrame into a Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # 3. Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(DIALOGUE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(DIALOGUE_MODEL_NAME)

    # 4. Set pad token to eos if pad_token is not defined (GPT-2 family doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    # 5. Define the map function for tokenization
    def map_fn(batch):
        return tokenize_function(batch, tokenizer)

    # 6. Tokenize the dataset
    tokenized_dataset = dataset.map(map_fn, batched=True)

    # 7. Set up training arguments
    #    report_to=[] ensures that wandb or other integrations won't be used
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        evaluation_strategy="no",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=500,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        report_to=[]  # Disable wandb or any other reporting integration
    )

    # 8. Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # 9. Fine-tune the model
    trainer.train()
    trainer.save_model(MODEL_OUTPUT_DIR)
    print("Dialogue model fine-tuning complete. Model saved to:", MODEL_OUTPUT_DIR)

if __name__ == "__main__":
    fine_tune_dialogue_model()

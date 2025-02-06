from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from src.config import DIALOGUE_MODEL_NAME, MODEL_OUTPUT_DIR

def tokenize_function(examples, tokenizer):
    inputs = examples["input_text"]
    targets = examples["target_text"]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding="max_length", truncation=True, max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def fine_tune_model():
    # Load the processed dataset from CSV
    dataset = load_dataset('csv', data_files={'train': 'data/processed/mental_health_train.csv'}, split='train')
    
    tokenizer = AutoTokenizer.from_pretrained(DIALOGUE_MODEL_NAME)
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    model = AutoModelForCausalLM.from_pretrained(DIALOGUE_MODEL_NAME)
    
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        evaluation_strategy="no",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=500,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    trainer.save_model(MODEL_OUTPUT_DIR)
    print("Fine-tuning complete. Model saved to:", MODEL_OUTPUT_DIR)

if __name__ == "__main__":
    fine_tune_model()

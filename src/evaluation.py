from transformers import AutoModelForCausalLM, AutoTokenizer
from config import DIALOGUE_MODEL_NAME, MODEL_OUTPUT_DIR, MAX_RESPONSE_LENGTH

def evaluate_response(user_input, history=""):
    tokenizer = AutoTokenizer.from_pretrained(DIALOGUE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_OUTPUT_DIR)
    input_text = user_input if not history else history + " " + user_input
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=MAX_RESPONSE_LENGTH,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Evaluate a sample input
    sample_input = "I feel extremely depressed and hopeless."
    print("Evaluation response:", evaluate_response(sample_input))

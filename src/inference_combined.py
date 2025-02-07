import os
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from config import (
    DIALOGUE_MODEL_NAME,
    MODEL_OUTPUT_DIR,
    MAX_RESPONSE_LENGTH,
    SENTIMENT_MODEL_NAME,
    SENTIMENT_MODEL_OUTPUT_DIR,
    CRISIS_SENTIMENT_THRESHOLD
)
import random

# ---------- Load KB.json (Knowledge Base) ----------
def load_kb(kb_path="data/raw/KB.json"):
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)
    intent_mapping = {}
    for intent in kb.get("intents", []):
        tag = intent.get("tag")
        responses = intent.get("responses", [])
        intent_mapping[tag] = responses
    return intent_mapping

# ---------- Load FAQ Dataset ----------
def load_faq(faq_path="data/raw/Mental_health_FAQ.csv"):
    faq_df = pd.read_csv(faq_path)
    faq_df = faq_df.rename(columns={"Questions": "question", "Answers": "answer"})
    return faq_df

# ---------- Load Fine-Tuned Dialogue Model ----------
def load_dialogue_model():
    tokenizer_dialogue = AutoTokenizer.from_pretrained(DIALOGUE_MODEL_NAME)
    model_dialogue = AutoModelForCausalLM.from_pretrained(MODEL_OUTPUT_DIR)
    return tokenizer_dialogue, model_dialogue

# ---------- Load Fine-Tuned Sentiment Model ----------
def load_sentiment_model():
    tokenizer_sentiment = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    model_sentiment = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_OUTPUT_DIR)
    return tokenizer_sentiment, model_sentiment

# ---------- Sentiment Analysis Function ----------
def analyze_sentiment(text):
    tokenizer_sentiment, model_sentiment = load_sentiment_model()
    inputs = tokenizer_sentiment(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model_sentiment(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    label_idx = torch.argmax(probabilities, dim=1).item()
    score = probabilities[0][label_idx].item()
    label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}.get(label_idx, "NEUTRAL")
    return {"label": label, "score": score}

# ---------- Dialogue Generation Function ----------
def generate_dialogue_response(user_input, history=""):
    tokenizer_dialogue, model_dialogue = load_dialogue_model()
    input_text = user_input if not history else history + " " + user_input
    input_ids = tokenizer_dialogue.encode(input_text + tokenizer_dialogue.eos_token, return_tensors="pt")
    output_ids = model_dialogue.generate(
        input_ids,
        max_length=MAX_RESPONSE_LENGTH,
        pad_token_id=tokenizer_dialogue.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    response = tokenizer_dialogue.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# ---------- FAQ Retrieval Function ----------
def retrieve_faq_response(user_input, faq_df, similarity_threshold=0.5):
    input_lower = user_input.lower()
    for index, row in faq_df.iterrows():
        question = str(row["question"]).lower()
        if any(word in input_lower for word in question.split()):
            return row["answer"]
    return None

# ---------- KB Intent Matching Function ----------
def match_kb_intent(user_input, kb_mapping):
    greetings = ["hi", "hey", "hello", "howdy", "hola", "bonjour", "konnichiwa", "namaste"]
    if any(word in user_input.lower() for word in greetings):
        responses = kb_mapping.get("greeting", [])
        if responses:
            return random.choice(responses)
    return None

# ---------- Crisis Management Function ----------
def crisis_response():
    return (
        "I'm really sorry that you're feeling this way. It appears you might need immediate help. "
        "If you're in crisis, please call your local emergency services or a crisis hotline immediately. "
        "For example, if you are in the United States, you can call 988 for crisis support. "
        "Please consider reaching out to someone you trust or a mental health professional."
    )

# ---------- Integrated Chatbot Function ----------
def chatbot_response(user_input, history=""):
    kb_mapping = load_kb("data/raw/KB.json")
    faq_df = load_faq("data/raw/Mental_health_FAQ.csv")
    
    kb_response = match_kb_intent(user_input, kb_mapping)
    if kb_response:
        return kb_response

    sentiment = analyze_sentiment(user_input)
    if sentiment["label"] == "NEGATIVE" and sentiment["score"] > CRISIS_SENTIMENT_THRESHOLD:
        return crisis_response()

    if "?" in user_input:
        faq_response = retrieve_faq_response(user_input, faq_df)
        if faq_response:
            return faq_response

    dialogue_response = generate_dialogue_response(user_input, history)
    return dialogue_response

# ---------- Main for Testing the Integrated Chatbot ----------
if __name__ == "__main__":
    print("Welcome to MINDCARE Chatbot!")
    history = ""
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Take care, and remember, you are not alone!")
            break
        response = chatbot_response(user_input, history)
        print("Chatbot:", response)
        history += " " + user_input + " " + response
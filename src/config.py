# Model names for dialogue generation and sentiment analysis
DIALOGUE_MODEL_NAME = "microsoft/DialoGPT-medium"
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Threshold for triggering crisis intervention (sentiment score)
CRISIS_SENTIMENT_THRESHOLD = 0.85

# Maximum token length for generated responses
MAX_RESPONSE_LENGTH = 100

# Paths for data and model storage
DATA_RAW_PATH = "data/raw/"
DATA_PROCESSED_PATH = "data/processed/"
MODEL_OUTPUT_DIR = "models/dialoGPT-finetuned/"
SENTIMENT_MODEL_OUTPUT_DIR = "models/sentiment-finetuned/"
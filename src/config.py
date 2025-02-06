# Model names and paths
DIALOGUE_MODEL_NAME = "microsoft/DialoGPT-medium"
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Crisis intervention threshold (sentiment score)
CRISIS_SENTIMENT_THRESHOLD = 0.85

# Maximum token length for generated responses
MAX_RESPONSE_LENGTH = 100

# Paths for data and model storage
DATA_RAW_PATH = "data/raw/crisis_text_line.csv"
DATA_PROCESSED_PATH = "data/processed/mental_health_train.csv"
MODEL_OUTPUT_DIR = "models/dialoGPT-finetuned/"
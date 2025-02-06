import pytest
from src.inference_combined import generate_dialogue_response, chatbot_response

def test_generate_dialogue_response():
    user_input = "I'm feeling very anxious."
    response = generate_dialogue_response(user_input)
    assert isinstance(response, str)
    assert len(response) > 0

def test_chatbot_response():
    user_input = "Hi there!"
    response = chatbot_response(user_input)
    assert isinstance(response, str)

if __name__ == "__main__":
    pytest.main()

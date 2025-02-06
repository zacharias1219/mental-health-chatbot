import pytest
from src.inference import generate_response

def test_generate_response():
    user_input = "I'm feeling very anxious."
    response = generate_response(user_input)
    assert isinstance(response, str)
    assert len(response) > 0

if __name__ == "__main__":
    pytest.main()

import sys
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# Fix import path before importing app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.main import app
from src.retriever.retriever import Retriever

# Mock the retriever for consistent responses
@pytest.fixture
def mock_retriever():
    # Create a mock Retriever instance
    mock = MagicMock(spec=Retriever)

    # Mock the get_similar_responses method to return predefined values
    mock.get_similar_responses.return_value = ["These are test responses"]
    
    # Inject the mock retriever into the FastAPI app state
    app.state.retriever = mock
    return mock

client = TestClient(app)

def test_query_endpoint(mock_retriever):
    response = client.post("/similar_responses", json={"question": "What is the capital of France?"})
    assert response.status_code == 200
    # Updated to check for 'results' instead of 'answers'
    assert response.json() == {"results": ["These are test responses"]}
    mock_retriever.get_similar_responses.assert_called_once_with("What is the capital of France?")

def test_blank_query_endpoint(mock_retriever):
    response = client.post("/similar_responses", json={"question": "?"})
    assert response.status_code == 200
    # Updated to check for 'results' instead of 'answers'
    assert response.json() == {"results": ["ERROR! Question input required!"]}

def test_whitespace_question(mock_retriever):
    response = client.post("/similar_responses", json={"question": "   "})
    assert response.status_code == 200
    # Updated to check for 'results' instead of 'answers'
    assert response.json() == {"results": ["ERROR! Question input required!"]}

def test_long_question(mock_retriever):
    long_question = "What is the meaning of life? " * 200
    response = client.post("/similar_responses", json={"question": long_question})
    assert response.status_code == 200
    # Updated to check for 'results' instead of 'answers'
    assert response.json() == {"results": ["Question is too long!"]}

def test_multilingual_question(mock_retriever):
    response = client.post("/similar_responses", json={"question": "Quelle est la capitale de la France?"})
    assert response.status_code == 200
    # Updated to check for 'results' instead of 'answers'
    assert response.json() == {"results": ["Question is not in English"]}


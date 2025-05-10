import pytest
from src.retriever.retriever import Retriever

@pytest.fixture(scope="module")
def retriever():
    return Retriever()

def test_data_loaded(retriever):
    assert retriever.df is not None
    assert len(retriever.df) > 0
    assert isinstance(retriever.texts, list)
    assert len(retriever.texts) == len(retriever.df)
    pass

def test_embeddings_created(retriever):
    assert retriever.embeddings is not None
    assert len(retriever.embeddings) == len(retriever.texts)
    pass

def test_get_similar_responses(retriever):
    question = "What is artificial intelligence?"
    results = retriever.get_similar_responses(question)
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(res, str) for res in results)
    pass

    



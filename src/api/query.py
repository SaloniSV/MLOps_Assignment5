from fastapi import APIRouter, Request, HTTPException
from src.models.query import RAGRequest, RAGResponse

router = APIRouter()

MAX_QUESTION_LENGTH = 500  # Define maximum length for a question

@router.post("/similar_responses", response_model=RAGResponse)
async def get_similar_responses(request_data: RAGRequest, request: Request):
    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not loaded")

    question = request_data.question.strip()

    # Check if the question is empty or just whitespace
    if not question:
        return {"results": ["ERROR! Question input required!"]}

    # Check for long questions
    if len(question) > MAX_QUESTION_LENGTH:
        return {"results": ["Question is too long!"]}

    # Proceed with retrieval if the question is valid
    try:
        results = retriever.get_similar_responses(question)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in retrieval: {str(e)}")
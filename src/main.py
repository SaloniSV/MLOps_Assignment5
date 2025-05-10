from fastapi import FastAPI
from src.api import query
from fastapi.responses import RedirectResponse
from src.models.query import RAGRequest, RAGResponse


app = FastAPI(
    title="ML API",
    description="API for ML Model Inference - Assignment 5 submitted by Saloni Vernekar",
    version="1.0.0",
)

@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# ✅ Add startup event to preload Retriever
@app.on_event("startup")
async def load_retriever():
    from src.retriever.retriever import Retriever  # Import here to avoid premature execution
    try:
        retriever = Retriever()
        app.state.retriever = retriever
        print("✅ Retriever loaded and embeddings are ready.")
    except Exception as e:
        print(f"❌ Error loading Retriever: {e}")
        app.state.retriever = None  # Fallback

# ✅ Include the router after startup is set up
app.include_router(query.router)





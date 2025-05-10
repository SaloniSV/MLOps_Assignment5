import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

class Retriever:
    def __init__(self):
        self.df = pd.read_csv("./data/6000_all_categories_questions_with_excerpts.csv")
        self.texts = self.df["wikipedia_excerpt"].tolist()

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_file = "./data/embeddings.pt"

        if os.path.exists(self.embedding_file):
            # 🔁 Load precomputed embeddings
            self.embeddings = torch.load(self.embedding_file)
            print("Loaded embeddings from disk.")
        else:
            # 🧠 Compute and save embeddings
            print("Computing embeddings...")
            self.embeddings = self.model.encode(self.texts, convert_to_tensor=True)
            torch.save(self.embeddings, self.embedding_file)
            print("Saved embeddings to disk.")

    def get_similar_responses(self, question: str, top_k: int = 3):
        query_embedding = self.model.encode(question, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        top_results = scores.topk(k=top_k)

        similar_texts = [self.texts[idx] for idx in top_results.indices]
        return similar_texts


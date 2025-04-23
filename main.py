# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import build_rag_chain

app = FastAPI()
rag_chain = build_rag_chain()

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "✅ RAG Chatbot API is running!"}

@app.post("/ask")
def ask_question(query: QueryRequest):
    try:
        answer = rag_chain.invoke(query.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

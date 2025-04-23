# rag_chain.py
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Connect to MongoDB
MONGO_URI = "mongodb+srv://anshshr:ansh123@freelancing-platform.esbya.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client["test"]
collection = db["machindetails"]

# Load documents
def load_documents():
    documents = []
    for doc in collection.find():
        text = "\n".join([f"{k}: {v}" for k, v in doc.items() if k != "_id"])
        documents.append(Document(page_content=text))
    return documents

# Build the RAG chain once and reuse
def build_rag_chain():
    docs = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="mongo_knowledge"
    )
    retriever = vectorstore.as_retriever()

    os.environ["GOOGLE_API_KEY"] = "AIzaSyAecRebjbiDdLuntjXDUXf8_QTAQbNscQw"
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    prompt = PromptTemplate.from_template("""
    You are a helpful assistant. Use the following context to answer the question:
    {context}

    Question: {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# backend/main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone
from llama_index.llms.openai import OpenAI
import os
import shutil
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone + embedding model
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(os.getenv("PINECONE_INDEX"))

# Embedding Model (for Pinecone)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Model for inference
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
Settings.llm=llm


vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def home():
    return {"status": 'ok', "message":"Nano-Notebook-LLM backend is working"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Load and index the PDF
    documents = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return {"status": "success", "filename": file.filename}


@app.post("/query")
async def query_doc(question: str = Form(...)):
    index = VectorStoreIndex.from_vector_store(vector_store)
    # Later we will update this to chat engine
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return {"response": str(response)}

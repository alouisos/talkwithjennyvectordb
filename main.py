from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple
import openai
import os
from dotenv import load_dotenv
from fastapi.responses import FileResponse
import json
from context import context
from context import context2
import faiss
import numpy as np
import re
import tiktoken
import os.path
import pickle
import hashlib

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure OpenAI
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No OPENAI_API_KEY found in environment variables")

client = openai.OpenAI(api_key=api_key)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

# Text preprocessing functions
def clean_text(text: str) -> str:
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s.]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    # Split text into overlapping chunks
    # chunk_size and overlap are in words
    print(f"\n=== Chunking Parameters ===")
    print(f"Chunk size: {chunk_size} words")
    print(f"Overlap: {overlap} words")
    
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Vector database setup
class VectorDB:
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        
    def save_to_disk(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'index': faiss.serialize_index(self.index), 'texts': self.texts}, f)
    
    def load_from_disk(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.index = faiss.deserialize_index(data['index'])
            self.texts = data['texts']

    def add_texts(self, texts: List[str]):
        embeddings = []
        for text in texts:
            embedding = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            ).data[0].embedding
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        self.texts.extend(texts)
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        query_embedding = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        ).data[0].embedding
        
        query_embedding_array = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding_array, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], distances[0][i]))
        return results

def get_context_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

# Initialize vector database
vector_db = VectorDB()

# Modify startup event to handle local storage
@app.on_event("startup")
async def startup_event():
    db_path = "vector_db.pkl"
    hash_path = "context_hash.txt"
    current_hash = get_context_hash(context2)
    
    # Check if we need to rebuild the database
    rebuild_db = True
    if os.path.exists(db_path) and os.path.exists(hash_path):
        with open(hash_path, 'r') as f:
            stored_hash = f.read().strip()
        if stored_hash == current_hash:
            try:
                vector_db.load_from_disk(db_path)
                rebuild_db = False
            except Exception as e:
                print(f"Error loading vector database: {e}")
    
    if rebuild_db:
        # Clean and chunk the context2
        cleaned_text = clean_text(context2)
        chunks = split_into_chunks(cleaned_text)
        
        # Add chunks to vector database
        vector_db.add_texts(chunks)
        
        # Save the database and hash
        vector_db.save_to_disk(db_path)
        with open(hash_path, 'w') as f:
            f.write(current_hash)

async def generate_response(messages: List[Message]):
    try:
        # Get the user's latest message
        user_message = messages[-1].content
        
        # Query vector database for relevant context
        relevant_chunks = vector_db.search(user_message, k=5)
        
        print("\n=== Additional Context from Vector DB ===")
        print("User query:", user_message)
        print("\nRelevant chunks (sorted by similarity):")
        for i, (chunk, similarity_score) in enumerate(relevant_chunks, 1):
            print(f"\nChunk {i}")
            print(f"Similarity score: {similarity_score:.4f}")
            print(f"Text: {chunk}")
        print("======================================\n")
        
        additional_context = "\n".join([chunk[0] for chunk in relevant_chunks])
        enhanced_context = f"{context}\n\nAdditional relevant information:\n{additional_context}"
        
        formatted_messages = [
            {'role': "system", "content": enhanced_context}
        ] + [{"role": msg.role, "content": msg.content} for msg in messages]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=formatted_messages,
            stream=True
        )
        
        for chunk in response:
            if chunk and chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(
        generate_response(request.messages),
        media_type="text/event-stream"
    )

@app.get("/")
async def root():
    return FileResponse('static/index.html')

# Add a health check endpoint
@app.get("/healthz")
async def health_check():
    return {"status": "healthy"} 
import io
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
import logging
# Configure logging to output to console with timestamps
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
# Create a logger for the service
logger = logging.getLogger("pdf_qa_service")
from fastapi import FastAPI, UploadFile, File, HTTPException

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from PyPDF2 import PdfReader

# Application configuration (environment variables)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")  # Qdrant service URL
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # Qdrant API key
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "my_pdf_collection")  # Vector store collection name
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")  # Ollama model name
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")  # Ollama server URL

app = FastAPI(title="PDF QA Service")  # FastAPI application instance
logger.info("Starting PDF QA Service")

# Initialize Qdrant client and collection
logger.info(f"Connecting to Qdrant at {QDRANT_URL}")
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
logger.info("Qdrant client initialized.")

# Check if the collection exists, create it if not
if not qdrant_client.collection_exists(COLLECTION_NAME):
    logger.info(f"Collection '{COLLECTION_NAME}' not found. Creating new collection.")
    try:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )
    except Exception as e:
        logger.error(f"Failed to create collection '{COLLECTION_NAME}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create collection: {e}")
else:
    logger.info(f"Collection '{COLLECTION_NAME}' already exists.")

# Initialize Embeddings and Vector Store
logger.info(f"Initializing embeddings with model '{OLLAMA_MODEL}' at {OLLAMA_URL}")
embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)
logger.info(f"Vector store initialized on collection '{COLLECTION_NAME}'.")


## Upload PDF and ingest document chunks into vector store
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    logger.info(f"Upload PDF endpoint called. Filename: {file.filename}")
    
    # Read the uploaded file
    data = await file.read()
    logger.info(f"Read {len(data)} bytes from uploaded file.")
    
    # Parse PDF and extract text
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() for page in reader.pages]
    logger.info(f"Extracted text from {len(pages)} PDF pages.")
    
    # Convert pages to Document objects
    docs = [Document(page_content=text) for text in pages if text]
    logger.info(f"Converted to {len(docs)} document chunks for ingestion.")
    
    # Ingest documents into Qdrant vector store
    if docs:
        vector_store.add_documents(docs)
        logger.info(f"Ingested {len(docs)} document chunks into vector store.")
    return {"ingested_chunks": len(docs)}


## Generate questions based on ingested PDF content
@app.get("/generate_questions")
async def generate_questions(k: int = 5, n: int = 4):
    logger.info(f"Generate questions endpoint called with k={k}, n={n}")
    # 1. Retrieve top-n chunks from Qdrant
    logger.info(f"Retrieving top {k} chunks from vector store.")
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # 2. Initialize Ollama LLM
    logger.info(f"Initializing LLM with model '{OLLAMA_MODEL}'.")
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
    )

    # 3. Build a prompt template for question generation
    logger.info(f"Building prompt for {n} questions.")
    prompt = PromptTemplate.from_template(
        f"Generate {n} insightful questions based on the following context:\n\n{{context}}"
    )

    # 4. Create chain to combine retrieved documents into prompt
    logger.info("Creating combine-documents chain.")
    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # 5. Wire retriever with combine-documents chain
    logger.info("Setting up retrieval chain.")
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_chain
    )

    # 6. Invoke retrieval chain to generate questions
    logger.info("Invoking retrieval chain.")
    output = retrieval_chain.invoke({"input": ""})
    answer = output.get("answer") or output.get("output") or str(output)
    logger.info(f"Raw answer from LLM: {answer}")

    # 7. Split the generated text into individual questions
    questions = [q.strip() for q in answer.splitlines() if q.strip()]
    logger.info(f"Split into {len(questions)} questions.")

    return {"questions": questions}

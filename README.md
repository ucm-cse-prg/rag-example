# RAG Example with FastAPI, Qdrant, and Ollama

A FastAPI-based application that ingests PDF documents into a Qdrant vector store and generates insightful questions using Ollama LLM embeddings.

## Features

- Upload PDF files and split into page-level chunks  
- Embed chunks with OllamaEmbeddings  
- Store embeddings in Qdrant  
- Retrieve top chunks and generate questions via OllamaLLM  

## Prerequisites

- Python 3.10+ (3.13 recommended) 
- Qdrant running (default http://localhost:6333)  
- Ollama server running (default http://localhost:11434)  

## Setup

1. Clone repository  
   ```bash
   git clone https://github.com/your-org/rag-example.git
   cd rag-example
   ```

2. Create & activate virtual environment  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Create a `.env` file in project root (see [main.py](main.py) for vars):  
   ```env
   QDRANT_URL=http://localhost:6333
   QDRANT_API_KEY=<your_api_key_if_any>
   COLLECTION_NAME=my_pdf_collection
   OLLAMA_MODEL=llama3.2
   OLLAMA_URL=http://localhost:11434
   ```

## Running the Service

Start FastAPI with Uvicorn:

```bash
uvicorn main:app --reload
```

By default it listens on `http://127.0.0.1:8000`.

## API Endpoints

### 1. Upload PDF

**POST** `/upload_pdf`  
- Form field: `file` (PDF)  

Example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/upload_pdf" \
  -F "file=@./example.pdf"
```

Response:

```json
{ "ingested_chunks": 10 }
```

### 2. Generate Questions

**GET** `/generate_questions`  
- Query params:  
  - `k` (number of questions, default 5)  
  - `n` (number of chunks to retrieve, default 4)  

Example:

```bash
curl "http://127.0.0.1:8000/generate_questions?k=5&n=4"
```

Response:

```json
{ "questions": [ "What is ...?", "How does ...?" ] }
```

## Logging

Logs are printed to the console with timestamps and levels. Adjust `logging.basicConfig` in [main.py](main.py) as needed.

## Troubleshooting

- **Qdrant not reachable**: ensure your service is up at `QDRANT_URL`.  
- **Ollama errors**: verify `OLLAMA_URL` and model names.  
- **Missing packages**: rerun `pip install -r requirements.txt`.  

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).  

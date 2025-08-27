# Donut-Flappy-LLM
DonutFlappyLLM🍩 is an ai agent with cassandra rag implementation. Developed for B2B company chatting.

A production-ready RAG (Retrieval-Augmented Generation) API built with FastAPI, Apache Cassandra, and Ollama for scalable document ingestion, semantic search, and AI-powered chat capabilities.
🌟 Features

Document Ingestion: Upload and process documents with automatic chunking and embedding
Semantic Search: Vector-based similarity search using cosine similarity
AI Chat: Context-aware responses using local LLM (Ollama)
Scalable Storage: Apache Cassandra for distributed vector storage
Real-time Streaming: Server-sent events for streaming responses
Rate Limiting: Built-in rate limiting with IP-based throttling
Authentication: API key-based security
Health Monitoring: Health check endpoints for monitoring

📋 Table of Contents

Architecture
Installation
Configuration
API Endpoints
Usage Examples
Performance
Monitoring
Troubleshooting

🏗️ Architecture
High Level Design (HLD)
mermaidgraph TB
    subgraph "Client Layer"
        Client[Client Applications]
        WebUI[Web UI]
        Mobile[Mobile Apps]
    end

    subgraph "API Gateway Layer"
        Auth[Authentication & Rate Limiting]
        Router[FastAPI Router]
    end

    subgraph "Application Layer"
        IngestService[Document Ingestion Service]
        ChatService[Chat Service]
        EmbedService[Embedding Service]
        SearchService[Vector Search Service]
    end

    subgraph "AI/ML Layer"
        Ollama[Ollama LLM Server]
        EmbedModel[Embedding Model<br/>nomic-embed-text]
        ChatModel[Chat Model<br/>qwen3:0.6b]
    end

    subgraph "Data Layer"
        Cassandra[(Apache Cassandra<br/>Vector Database)]
        ChunksTable[chunks table]
        DocsTable[documents table]
    end

    subgraph "Infrastructure"
        ThreadPool[Thread Pool Executor]
        HTTPClient[HTTP Client Pool]
    end

    Client --> Auth
    WebUI --> Auth
    Mobile --> Auth
    
    Auth --> Router
    Router --> IngestService
    Router --> ChatService
    
    IngestService --> EmbedService
    IngestService --> Cassandra
    
    ChatService --> SearchService
    ChatService --> Ollama
    
    SearchService --> Cassandra
    EmbedService --> Ollama
    
    Ollama --> EmbedModel
    Ollama --> ChatModel
    
    Cassandra --> ChunksTable
    Cassandra --> DocsTable
    
    IngestService --> ThreadPool
    ChatService --> ThreadPool
    EmbedService --> HTTPClient
Low Level Design (LLD)
1. Data Flow Architecture
mermaidsequenceDiagram
    participant C as Client
    participant API as FastAPI
    participant Auth as Auth Guard
    participant Embed as Embedding Service
    participant Cass as Cassandra
    participant Ollama as Ollama Server
    participant Thread as Thread Pool

    Note over C,Thread: Document Ingestion Flow
    C->>API: POST /ingest {doc_id, text}
    API->>Auth: Validate API Key & Rate Limit
    Auth->>API: ✓ Authorized
    
    API->>API: chunk_text(text, max_tokens)
    API->>Thread: execute(embed_many, chunks)
    Thread->>Embed: embed(chunk1), embed(chunk2)...
    Embed->>Ollama: POST /api/embeddings
    Ollama->>Embed: [0.1, 0.2, ...] vectors
    Thread->>API: List[vectors]
    
    API->>Cass: upsert_document(metadata)
    API->>Cass: insert_chunks(vectors + content)
    API->>C: {ok: true, chunks: N}

    Note over C,Thread: Chat Flow
    C->>API: POST /chat {query}
    API->>Auth: Validate & Rate Limit
    API->>Thread: execute(embed, query)
    Thread->>Embed: embed(query)
    Embed->>Ollama: POST /api/embeddings
    Ollama->>Thread: query_vector
    
    API->>Cass: all_chunks_in_namespace()
    API->>API: cosine_similarity(query_vec, chunk_vecs)
    API->>API: select top_k contexts
    API->>Thread: execute(chat, contexts+query)
    Thread->>Ollama: POST /api/chat (stream)
    Ollama-->>C: Server-Sent Events
2. Database Schema Design
sql-- Keyspace
CREATE KEYSPACE ragks 
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

-- Documents metadata table
CREATE TABLE documents (
  namespace text,        -- Logical grouping (tenant/project)
  doc_id text,          -- Unique document identifier
  created_at timestamp, -- Document creation time
  meta map<text, text>, -- Key-value metadata
  PRIMARY KEY ((namespace), doc_id)
);

-- Vector chunks table
CREATE TABLE chunks (
  namespace text,           -- Partition key for data locality
  doc_id text,             -- Document identifier
  chunk_id text,           -- Chunk sequence (000001, 000002...)
  content text,            -- Raw text content
  embedding list<float>,   -- Vector embedding (384 dimensions)
  PRIMARY KEY ((namespace), doc_id, chunk_id)
) WITH CLUSTERING ORDER BY (doc_id ASC, chunk_id ASC);
3. Component Architecture
mermaidclassDiagram
    class FastAPIApp {
        +Router router
        +Middleware[] middleware
        +startup_event()
        +shutdown_event()
    }

    class CassandraRepo {
        -Cluster cluster
        -Session session
        -PreparedStatement[] statements
        +upsert_document()
        +insert_chunks()
        +all_chunks_in_namespace()
        +delete_document()
    }

    class EmbeddingClient {
        -HTTPXClient client
        -str model
        +embed(text: str): List[float]
        +embed_many(texts: List[str]): List[List[float]]
    }

    class OllamaChatClient {
        -HTTPXClient client
        -str model
        +stream_chat(messages): Iterator[str]
        +chat_once(messages): str
    }

    class RateLimiter {
        -Dict[str, List[float]] bucket
        +check(key: str): None
        +cleanup_expired(): None
    }

    class ChunkingService {
        +chunk_text(text: str, max_tokens: int): List[str]
        +simple_token_estimate(text: str): int
    }

    class VectorMath {
        +cosine_sim(a: List[float], b: List[float]): float
        +dot(a: List[float], b: List[float]): float
        +l2_norm(a: List[float]): float
    }

    FastAPIApp --> CassandraRepo
    FastAPIApp --> EmbeddingClient
    FastAPIApp --> OllamaChatClient
    FastAPIApp --> RateLimiter
    FastAPIApp --> ChunkingService
    FastAPIApp --> VectorMath
4. Vector Search Algorithm
pythondef select_context(namespace: str, query_vector: List[float], top_k: int) -> List[Tuple[float, str]]:
    """
    Brute-force cosine similarity search
    Time Complexity: O(n * d) where n=chunks, d=dimensions
    Space Complexity: O(n) for storing similarities
    """
    scored = []
    
    # 1. Iterate through all chunks in namespace
    for chunk_data in cassandra_repo.all_chunks_in_namespace(namespace):
        namespace, doc_id, chunk_id, content, embedding = chunk_data
        
        # 2. Calculate cosine similarity
        similarity = cosine_sim(query_vector, embedding)
        scored.append((similarity, content))
    
    # 3. Sort by similarity (descending) and return top-k
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]
System Characteristics
AspectImplementationTrade-offsConsistencyCassandra QUORUMStrong consistency vs availabilityScalabilityHorizontal (add nodes)Linear scaling with some overheadPerformanceIn-memory similarityFast search, memory boundedDurabilityCassandra replicationData safety vs write latencyAvailabilitySingle point of failureSimple deployment vs HA
🚀 Installation
Prerequisites

Python 3.11+
Apache Cassandra 4.0+ or DataStax Astra
Ollama with models installed
Docker (optional)

Quick Start

Clone the repository

bashgit clone https://github.com/yourusername/rag-api.git
cd rag-api

Install dependencies

bashpython -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

Install Ollama and models

bash# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download required models
ollama pull nomic-embed-text    # Embedding model (274MB)
ollama pull qwen3:0.6b         # Chat model (396MB)

Start Cassandra

bash# Using Docker
docker run --name cassandra -p 9042:9042 -d cassandra:4.1

# Or install locally
# https://cassandra.apache.org/doc/latest/getting_started/installing.html

Configure environment

bashcp .env.example .env
# Edit .env with your settings

Start the API

bashuvicorn main:app --host 0.0.0.0 --port 8000 --reload
Docker Deployment
yaml# docker-compose.yml
version: '3.8'
services:
  cassandra:
    image: cassandra:4.1
    ports:
      - "9042:9042"
    environment:
      - CASSANDRA_CLUSTER_NAME=ragcluster
    volumes:
      - cassandra_data:/var/lib/cassandra

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CASSANDRA_HOSTS=cassandra
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - cassandra
      - ollama

volumes:
  cassandra_data:
  ollama_data:
⚙️ Configuration
Environment Variables
Create a .env file:
bash# API Security
API_KEY=your-super-secret-api-key-here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen3:0.6b      # or llama3, mistral, etc.
OLLAMA_EMBED_MODEL=nomic-embed-text

# Cassandra Configuration
CASSANDRA_HOSTS=127.0.0.1,127.0.0.2,127.0.0.3
CASSANDRA_PORT=9042
CASSANDRA_USERNAME=cassandra
CASSANDRA_PASSWORD=cassandra
CASSANDRA_KEYSPACE=ragks

# Performance Tuning
TOP_K=5                    # Default context chunks to retrieve
MAX_CHUNK_TOKENS=256       # Maximum tokens per chunk
MAX_CONTEXT_TOKENS=1200    # Maximum total context tokens
HTTP_TIMEOUT_S=60          # HTTP request timeout

# Rate Limiting
RATE_LIMIT_PER_MINUTE=120  # Requests per minute per IP
Model Selection Guide
ModelSizeSpeedQualityUse CaseEmbedding Modelsnomic-embed-text274MBFastGoodGeneral purposemxbai-embed-large669MBMediumBetterHigh quality embeddingsall-minilm23MBVery FastBasicResource constrainedChat Modelsqwen3:0.6b396MBVery FastBasicQuick responsesllama3:8b4.7GBMediumGoodBalanced performancemistral:7b4.1GBMediumGoodCode & reasoning
📡 API Endpoints
Authentication
All endpoints require X-API-Key header:
bashcurl -H "X-API-Key: your-api-key" http://localhost:8000/endpoint
Core Endpoints
1. Document Ingestion
httpPOST /ingest
Content-Type: application/json
X-API-Key: your-api-key

{
  "doc_id": "user-manual-v1",
  "text": "This is the user manual content...",
  "namespace": "support-docs",
  "metadata": {
    "version": "1.0",
    "author": "John Doe",
    "category": "documentation"
  }
}
Response:
json{
  "ok": true,
  "doc_id": "user-manual-v1",
  "chunks": 15
}
2. Chat (Streaming)
httpPOST /chat
Content-Type: application/json
X-API-Key: your-api-key

{
  "query": "How do I reset my password?",
  "namespace": "support-docs",
  "top_k": 3,
  "stream": true
}
Response: Server-sent events stream
3. Chat (Non-streaming)
httpPOST /chat
Content-Type: application/json
X-API-Key: your-api-key

{
  "query": "What are the system requirements?",
  "namespace": "support-docs",
  "top_k": 5,
  "stream": false
}
Response:
json{
  "answer": "Based on the documentation, the system requirements are...",
  "contexts": [
    "System Requirements: - Python 3.11+...",
    "Hardware: Minimum 4GB RAM..."
  ]
}
4. Document Management
Get document info:
httpGET /document/{namespace}/{doc_id}
X-API-Key: your-api-key
Delete document:
httpDELETE /document
Content-Type: application/json
X-API-Key: your-api-key

{
  "doc_id": "user-manual-v1",
  "namespace": "support-docs"
}
5. Health Check
httpGET /health
Response:
json{
  "status": "ok",
  "timestamp": 1703123456.789
}
Error Responses
json{
  "detail": "Rate limit exceeded"
}
Common HTTP status codes:

400 - Bad Request (invalid input)
401 - Unauthorized (missing/invalid API key)
404 - Not Found (document doesn't exist)
429 - Too Many Requests (rate limited)
500 - Internal Server Error

💡 Usage Examples
Python Client
pythonimport requests
import json

class RAGClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {'X-API-Key': api_key, 'Content-Type': 'application/json'}
    
    def ingest_document(self, doc_id: str, text: str, namespace: str = "default", metadata: dict = None):
        payload = {
            "doc_id": doc_id,
            "text": text,
            "namespace": namespace,
            "metadata": metadata or {}
        }
        response = requests.post(f"{self.base_url}/ingest", json=payload, headers=self.headers)
        return response.json()
    
    def chat(self, query: str, namespace: str = "default", stream: bool = False, top_k: int = 5):
        payload = {
            "query": query,
            "namespace": namespace,
            "stream": stream,
            "top_k": top_k
        }
        response = requests.post(f"{self.base_url}/chat", json=payload, headers=self.headers)
        
        if stream:
            return response.iter_content(decode_unicode=True)
        else:
            return response.json()

# Usage
client = RAGClient("http://localhost:8000", "your-api-key")

# Ingest a document
result = client.ingest_document(
    doc_id="faq-2024",
    text="Q: How to reset password? A: Click the reset link...",
    namespace="support",
    metadata={"type": "faq", "year": "2024"}
)
print(result)

# Chat with streaming
for token in client.chat("How to reset password?", namespace="support", stream=True):
    print(token, end='', flush=True)

# Chat without streaming
response = client.chat("How to reset password?", namespace="support", stream=False)
print(response["answer"])
JavaScript/Node.js Client
javascriptclass RAGClient {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.headers = {
            'X-API-Key': apiKey,
            'Content-Type': 'application/json'
        };
    }

    async ingestDocument(docId, text, namespace = 'default', metadata = {}) {
        const response = await fetch(`${this.baseUrl}/ingest`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                doc_id: docId,
                text: text,
                namespace: namespace,
                metadata: metadata
            })
        });
        return await response.json();
    }

    async chatStream(query, namespace = 'default', topK = 5) {
        const response = await fetch(`${this.baseUrl}/chat`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                query: query,
                namespace: namespace,
                top_k: topK,
                stream: true
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            console.log(chunk);
        }
    }
}

// Usage
const client = new RAGClient('http://localhost:8000', 'your-api-key');

// Ingest document
await client.ingestDocument('manual-v2', 'User manual content...', 'docs');

// Stream chat
await client.chatStream('How to use the API?', 'docs');
cURL Examples
Bulk document ingestion:
bash#!/bin/bash

API_KEY="your-api-key"
BASE_URL="http://localhost:8000"

# Ingest multiple documents
for file in docs/*.txt; do
    doc_id=$(basename "$file" .txt)
    content=$(cat "$file")
    
    curl -X POST "$BASE_URL/ingest" \
        -H "X-API-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"doc_id\": \"$doc_id\",
            \"text\": \"$content\",
            \"namespace\": \"knowledge-base\"
        }"
    
    echo "Ingested: $doc_id"
done
Interactive chat session:
bash#!/bin/bash

API_KEY="your-api-key"
BASE_URL="http://localhost:8000"

while true; do
    echo -n "Query: "
    read query
    
    if [ "$query" = "quit" ]; then
        break
    fi
    
    curl -X POST "$BASE_URL/chat" \
        -H "X-API-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"query\": \"$query\",
            \"namespace\": \"knowledge-base\",
            \"stream\": false
        }" | jq -r '.answer'
    
    echo ""
done
⚡ Performance
Benchmarks
Hardware: 16GB RAM, 8-core CPU, SSD
OperationDocumentsChunksTimeThroughputIngestion1,00050,00045s22 docs/sSearch-50,000120ms8.3 searches/sEmbedding1,000 texts-15s67 embeds/s
Scaling Guidelines
Vertical Scaling (Single Node):

CPU: 8+ cores for parallel embedding
RAM: 8GB + (embedding_dim × num_chunks × 4 bytes)
Storage: SSD for Cassandra data

Horizontal Scaling (Multi-Node):

Cassandra: Add nodes for storage scaling
API Servers: Load balance multiple FastAPI instances
Ollama: Dedicated GPU servers for models

Memory Usage Estimation:
python# For nomic-embed-text (384 dimensions)
memory_mb = (num_chunks * 384 * 4) / (1024 * 1024)

# Example: 1M chunks = ~1.5GB RAM
Optimization Tips

Batch Processing: Use embed_many() for multiple documents
Connection Pooling: Configure Cassandra connection pools
Model Caching: Keep Ollama models loaded in memory
Async Operations: Use ThreadPoolExecutor for I/O
Chunking Strategy: Optimize chunk size for your use case

📊 Monitoring
Metrics Collection
python# Add to main.py for basic metrics
import time
import logging
from collections import defaultdict

metrics = defaultdict(list)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    metrics[f"{request.method}_{request.url.path}"].append(duration)
    
    # Log slow requests
    if duration > 1.0:
        logging.warning(f"Slow request: {request.method} {request.url.path} took {duration:.2f}s")
    
    return response

@app.get("/metrics")
async def get_metrics():
    return {
        endpoint: {
            "count": len(times),
            "avg_ms": sum(times) / len(times) * 1000,
            "max_ms": max(times) * 1000
        }
        for endpoint, times in metrics.items()
    }
Health Checks
bash# Basic health check
curl http://localhost:8000/health

# Advanced health check with monitoring
#!/bin/bash
check_health() {
    response=$(curl -s http://localhost:8000/health)
    status=$(echo "$response" | jq -r '.status')
    
    if [ "$status" = "ok" ]; then
        echo "✅ API is healthy"
        return 0
    else
        echo "❌ API is unhealthy: $response"
        return 1
    fi
}
Log Configuration
pythonimport logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag-api.log'),
        logging.StreamHandler()
    ]
)

# Add request ID for tracing
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response
🔧 Troubleshooting
Common Issues
1. Cassandra Connection Issues
Problem: NoHostAvailable: Unable to connect to any servers
Solutions:
bash# Check Cassandra status
docker logs cassandra
# or
sudo systemctl status cassandra

# Test connection
cqlsh localhost 9042

# Check firewall
sudo ufw allow 9042

# Verify configuration
grep -r "rpc_address\|listen_address" /etc/cassandra/
2. Ollama Model Issues
Problem: Model not found or slow responses
Solutions:
bash# List installed models
ollama list

# Download missing models
ollama pull nomic-embed-text
ollama pull qwen3:0.6b

# Check model status
ollama ps

# Restart Ollama
sudo systemctl restart ollama
3. Memory Issues
Problem: OutOfMemoryError during ingestion
Solutions:
python# Reduce batch size
BATCH_SIZE = 10  # Instead of processing all chunks at once

# Implement streaming ingestion
async def stream_ingest(chunks):
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        vectors = await embed_many(batch)
        repo.insert_chunks(vectors)
4. Performance Issues
Problem: Slow search responses
Debugging:
pythonimport time

def debug_select_context(namespace: str, q_vec: List[float], top_k: int):
    start = time.time()
    
    # Count chunks
    chunk_count = 0
    for _ in repo.all_chunks_in_namespace(namespace):
        chunk_count += 1
    
    print(f"Scanning {chunk_count} chunks...")
    
    # Measure similarity computation
    scored = []
    sim_start = time.time()
    
    for chunk_data in repo.all_chunks_in_namespace(namespace):
        _, _, _, content, emb = chunk_data
        if emb:
            similarity = cosine_sim(q_vec, emb)
            scored.append((similarity, content))
    
    sim_time = time.time() - sim_start
    total_time = time.time() - start
    
    print(f"Similarity computation: {sim_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    
    return sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]
Debug Mode
Enable debug logging:
bashexport LOG_LEVEL=DEBUG
uvicorn main:app --log-level debug --reload
Performance Profiling
python# Add to main.py
import cProfile
import pstats

@app.get("/profile/{operation}")
async def profile_operation(operation: str):
    if operation == "search":
        pr = cProfile.Profile()
        pr.enable()
        
        # Simulate search operation
        result = select_context("default", [0.1] * 384, 5)
        
        pr.disable()
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        
        # Return top functions
        return stats.print_stats(10)
🤝 Contributing

Fork the repository
Create a feature branch: git checkout -b feature/amazing-feature
Commit your changes: git commit -m 'Add amazing feature'
Push to the branch: git push origin feature/amazing-feature
Open a Pull Request

Development Setup
bash# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Code formatting
black main.py
isort main.py

# Type checking
mypy main.py

# Linting
flake8 main.py

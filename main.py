from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Iterable, List, Optional, Sequence, Tuple

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement, BatchStatement, ConsistencyLevel

# ----------------------------- Config & Utilities -----------------------------

@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("API_KEY", "changeme")
    ollama_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:0.6b")
    embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    cass_hosts: Tuple[str, ...] = tuple(os.getenv("CASSANDRA_HOSTS", "127.0.0.1").split(","))
    cass_port: int = int(os.getenv("CASSANDRA_PORT", "9042"))
    cass_user: str = os.getenv("CASSANDRA_USERNAME", "cassandra")
    cass_pass: str = os.getenv("CASSANDRA_PASSWORD", "cassandra")
    keyspace: str = os.getenv("CASSANDRA_KEYSPACE", "ragks")

    top_k: int = int(os.getenv("TOP_K", "5"))
    max_chunk_tokens: int = int(os.getenv("MAX_CHUNK_TOKENS", "256"))
    max_context_tokens: int = int(os.getenv("MAX_CONTEXT_TOKENS", "1200"))
    request_timeout_s: int = int(os.getenv("HTTP_TIMEOUT_S", "60"))

SETTINGS = Settings()

# Basic in-memory rate limiter (best-effort) -----------------------------------
class RateLimiter:
    def __init__(self, max_per_minute: int = 120) -> None:
        self.max_per_minute = max_per_minute
        self._bucket: dict[str, List[float]] = {}

    def check(self, key: str) -> None:
        now = time.time()
        window_start = now - 60
        q = self._bucket.setdefault(key, [])
        while q and q[0] < window_start:
            q.pop(0)
        if len(q) >= self.max_per_minute:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        q.append(now)

rate_limiter = RateLimiter()

# Security dependency -----------------------------------------------------------
async def auth_guard(req: Request) -> None:
    sent = req.headers.get("X-API-Key")
    if not sent or sent != SETTINGS.api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # best-effort rate limit by client ip
    client_ip = req.client.host if req.client else "unknown"
    rate_limiter.check(client_ip)

# --------------------------- Cassandra Repository -----------------------------

class CassandraRepo:
    def __init__(self) -> None:
        auth = PlainTextAuthProvider(SETTINGS.cass_user, SETTINGS.cass_pass)
        self.cluster = Cluster(SETTINGS.cass_hosts, port=SETTINGS.cass_port, auth_provider=auth)
        self.session = self.cluster.connect()
        self._init_schema()
        # Prepared statements for better performance and reliability
        self._select_chunks_stmt = None
        self._insert_chunk_stmt = None
        self._select_document_stmt = None
        self._prepare_statements()

    def _init_schema(self) -> None:
        ks = SETTINGS.keyspace
        self.session.execute(
            f"""
            CREATE KEYSPACE IF NOT EXISTS {ks}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}};
            """
        )
        self.session.set_keyspace(ks)
        # store embeddings as list<float> for broad compatibility
        self.session.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              namespace text,
              doc_id text,
              chunk_id text,
              content text,
              embedding list<float>,
              PRIMARY KEY ((namespace), doc_id, chunk_id)
            ) WITH CLUSTERING ORDER BY (doc_id ASC, chunk_id ASC);
            """
        )
        # metadata table for documents
        self.session.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
              namespace text,
              doc_id text,
              created_at timestamp,
              meta map<text, text>,
              PRIMARY KEY ((namespace), doc_id)
            );
            """
        )

    def _prepare_statements(self) -> None:
        """Prepare frequently used statements for better performance"""
        self._select_chunks_stmt = self.session.prepare(
            "SELECT namespace, doc_id, chunk_id, content, embedding FROM chunks WHERE namespace = ?"
        )
        self._insert_chunk_stmt = self.session.prepare(
            "INSERT INTO chunks (namespace, doc_id, chunk_id, content, embedding) VALUES (?, ?, ?, ?, ?)"
        )
        self._select_document_stmt = self.session.prepare(
            "SELECT namespace, doc_id, created_at, meta FROM documents WHERE namespace = ? AND doc_id = ?"
        )

    # --------------------------- Write Paths ----------------------------------
    def upsert_document(self, namespace: str, doc_id: str, meta: Optional[dict[str, str]] = None) -> None:
        stmt = SimpleStatement(
            "INSERT INTO documents (namespace, doc_id, created_at, meta) VALUES (?, ?, toTimestamp(now()), ?) IF NOT EXISTS",
            consistency_level=ConsistencyLevel.QUORUM,
        )
        self.session.execute(stmt, [namespace, doc_id, meta or {}])

    def insert_chunks(self, rows: Sequence[Tuple[str, str, str, str, List[float]]]) -> None:
        bs = BatchStatement(consistency_level=ConsistencyLevel.QUORUM)
        for row in rows:
            bs.add(self._insert_chunk_stmt, row)
        self.session.execute(bs)

    # --------------------------- Read Paths -----------------------------------
    def all_chunks_in_namespace(self, namespace: str) -> Iterable[Tuple[str, str, str, str, List[float]]]:
        # Use prepared statement with list parameter to avoid tuple formatting issues
        rs = self.session.execute(self._select_chunks_stmt, [namespace])
        for r in rs:
            yield (r.namespace, r.doc_id, r.chunk_id, r.content, r.embedding)

    def get_document(self, namespace: str, doc_id: str) -> Optional[dict]:
        """Get document metadata"""
        rs = self.session.execute(self._select_document_stmt, [namespace, doc_id])
        row = rs.one()
        if row:
            return {
                'namespace': row.namespace,
                'doc_id': row.doc_id,
                'created_at': row.created_at,
                'meta': dict(row.meta) if row.meta else {}
            }
        return None

    def delete_document_chunks(self, namespace: str, doc_id: str) -> None:
        """Delete all chunks for a document"""
        stmt = SimpleStatement(
            "DELETE FROM chunks WHERE namespace = ? AND doc_id = ?",
            consistency_level=ConsistencyLevel.QUORUM
        )
        self.session.execute(stmt, [namespace, doc_id])

    def delete_document(self, namespace: str, doc_id: str) -> None:
        """Delete document metadata"""
        stmt = SimpleStatement(
            "DELETE FROM documents WHERE namespace = ? AND doc_id = ?",
            consistency_level=ConsistencyLevel.QUORUM
        )
        self.session.execute(stmt, [namespace, doc_id])

repo = CassandraRepo()

# ------------------------------ Embeddings ------------------------------------

class EmbeddingClient:
    def __init__(self, base_url: str, model: str, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def embed(self, text: str) -> List[float]:
        # normalize text
        text = text.strip()
        if not text:
            raise ValueError("Empty text cannot be embedded")
            
        payload = {"model": self.model, "prompt": text}
        try:
            resp = self._client.post(f"{self.base_url}/api/embeddings", json=payload)
            resp.raise_for_status()
            data = resp.json()
            vec = data.get("embedding")
            if not isinstance(vec, list):
                raise RuntimeError(f"Unexpected embedding response: {data}")
            return [float(x) for x in vec]
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to get embedding: {e}")

    def __del__(self):
        """Clean up HTTP client"""
        try:
            self._client.close()
        except:
            pass

embedding_client = EmbeddingClient(SETTINGS.ollama_url, SETTINGS.embed_model, SETTINGS.request_timeout_s)

# ------------------------------ Chunking --------------------------------------

def simple_token_estimate(text: str) -> int:
    # crude token estimate ~4 chars/token
    return max(1, math.ceil(len(text) / 4))

def chunk_text(text: str, max_tokens: int) -> List[str]:
    # whitespace-aware greedy chunker
    if not text.strip():
        return []
        
    words = text.split()
    if not words:
        return []
        
    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0
    
    for w in words:
        t = simple_token_estimate(w + " ")
        if cur_tokens + t > max_tokens and cur:
            chunks.append(" ".join(cur).strip())
            cur, cur_tokens = [], 0
        cur.append(w)
        cur_tokens += t
        
    if cur:
        chunks.append(" ".join(cur).strip())
        
    return [c for c in chunks if c.strip()]  # Filter out empty chunks

# ------------------------------ Math utils ------------------------------------

def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    return float(sum(x*y for x, y in zip(a, b)))

def l2(a: Sequence[float]) -> float:
    return math.sqrt(sum(x*x for x in a))

def cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    da, db = l2(a), l2(b)
    if da == 0 or db == 0:
        return 0.0
    return dot(a, b) / (da * db)

# ------------------------------ Pydantic --------------------------------------

class IngestReq(BaseModel):
    doc_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    namespace: str = Field("default", min_length=1)
    metadata: Optional[dict[str, str]] = None

class ChatReq(BaseModel):
    query: str = Field(..., min_length=1)
    namespace: str = Field("default", min_length=1)
    top_k: Optional[int] = Field(None, ge=1, le=50)
    stream: bool = Field(True)
    session_id: Optional[str] = None

class DeleteReq(BaseModel):
    doc_id: str = Field(..., min_length=1)
    namespace: str = Field("default", min_length=1)

# ----------------------------- RAG Core ---------------------------------------

def select_context(namespace: str, q_vec: List[float], top_k: int) -> List[Tuple[float, str]]:
    # brute-force cosine in Python (works anywhere). For scale, replace with server-side ANN.
    scored: List[Tuple[float, str]] = []
    
    try:
        for _ns, _doc, _chunk, content, emb in repo.all_chunks_in_namespace(namespace):
            if not emb or not content.strip():
                continue
            try:
                s = cosine_sim(q_vec, emb)
                scored.append((s, content))
            except Exception as e:
                # Log embedding comparison error but continue
                print(f"Warning: Failed to compute similarity for chunk: {e}")
                continue
                
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]
    except Exception as e:
        print(f"Error in select_context: {e}")
        return []

SYSTEM_GUARDRAILS = (
    "You are a concise, careful assistant. Use the provided context snippets to answer. "
    "If the answer is not in the context, say you don't know. "
    "Never invent sources. Keep answers short and factual."
)

def build_prompt(contexts: List[str], question: str) -> List[dict]:
    if not contexts:
        context_block = "[No relevant context found]"
    else:
        context_block = "\n\n".join(f"[CTX {i+1}]\n{c}" for i, c in enumerate(contexts))
    
    return [
        {"role": "system", "content": SYSTEM_GUARDRAILS},
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
    ]

# ----------------------------- Ollama Chat ------------------------------------

class OllamaChatClient:
    def __init__(self, base_url: str, model: str, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def stream_chat(self, messages: List[dict]) -> Iterable[str]:
        payload = {"model": self.model, "messages": messages, "stream": True}
        try:
            with self._client.stream("POST", f"{self.base_url}/api/chat", json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to stream chat: {e}")

    def chat_once(self, messages: List[dict]) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False}
        try:
            resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to get chat response: {e}")

    def __del__(self):
        """Clean up HTTP client"""
        try:
            self._client.close()
        except:
            pass

chat_client = OllamaChatClient(SETTINGS.ollama_url, SETTINGS.chat_model, SETTINGS.request_timeout_s)

# ThreadPool for parallel embeddings ------------------------------------------
EMBED_POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

def embed_many(texts: Sequence[str]) -> List[List[float]]:
    """Embed multiple texts in parallel"""
    if not texts:
        return []
    fn = partial(embedding_client.embed)
    return list(EMBED_POOL.map(fn, texts))

# ------------------------------- FastAPI --------------------------------------

app = FastAPI(title="RAG API (Ollama + Cassandra)")

@app.post("/ingest", dependencies=[Depends(auth_guard)])
async def ingest(req: IngestReq):
    namespace = req.namespace.strip()
    doc_id = req.doc_id.strip()
    text = req.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text content is required")

    try:
        # idempotent doc row
        repo.upsert_document(namespace, doc_id, (req.metadata or {}))

        # chunk & embed concurrently
        chunks = chunk_text(text, SETTINGS.max_chunk_tokens)
        if not chunks:
            raise HTTPException(status_code=400, detail="No content after chunking")

        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(EMBED_POOL, embed_many, chunks)

        rows = []
        for i, (c, v) in enumerate(zip(chunks, vectors)):
            rows.append((namespace, doc_id, f"{i:06d}", c, v))

        repo.insert_chunks(rows)

        return {"ok": True, "doc_id": doc_id, "chunks": len(rows)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/chat", dependencies=[Depends(auth_guard)])
async def chat(req: ChatReq):
    namespace = req.namespace.strip()
    q = req.query.strip()
    top_k = req.top_k or SETTINGS.top_k

    if not q:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        # embed query
        loop = asyncio.get_event_loop()
        q_vec = await loop.run_in_executor(EMBED_POOL, embedding_client.embed, q)

        # select contexts
        top = select_context(namespace, q_vec, top_k)
        contexts = [c for _, c in top]

        messages = build_prompt(contexts, q)

        if req.stream:
            async def token_gen():
                try:
                    for tok in chat_client.stream_chat(messages):
                        yield tok
                except Exception as e:
                    yield f"Error: {str(e)}"
            return StreamingResponse(token_gen(), media_type="text/plain")
        else:
            out = await loop.run_in_executor(EMBED_POOL, chat_client.chat_once, messages)
            return JSONResponse({"answer": out, "contexts": contexts})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.delete("/document", dependencies=[Depends(auth_guard)])
async def delete_document(req: DeleteReq):
    """Delete a document and all its chunks"""
    namespace = req.namespace.strip()
    doc_id = req.doc_id.strip()
    
    try:
        # Check if document exists
        doc = repo.get_document(namespace, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete chunks first, then document metadata
        repo.delete_document_chunks(namespace, doc_id)
        repo.delete_document(namespace, doc_id)
        
        return {"ok": True, "message": f"Document {doc_id} deleted"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@app.get("/document/{namespace}/{doc_id}", dependencies=[Depends(auth_guard)])
async def get_document_info(namespace: str, doc_id: str):
    """Get document metadata"""
    try:
        doc = repo.get_document(namespace.strip(), doc_id.strip())
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@app.get("/health")
async def health():
    try:
        repo.session.execute("SELECT now() FROM system.local")
        return {"status": "ok", "timestamp": time.time()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "fail", "error": str(e)})

# ---------------------------- Graceful shutdown -------------------------------

@app.on_event("shutdown")
async def shutdown_event():
    try:
        EMBED_POOL.shutdown(wait=False, cancel_futures=True)
    finally:
        try:
            repo.cluster.shutdown()
        except Exception:
            pass
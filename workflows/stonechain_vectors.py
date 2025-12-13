#!/usr/bin/env python3
"""
StoneChain Vector Database Integrations
=======================================

Optional integrations for production vector databases.
These require external dependencies but follow StoneChain's philosophy:
minimal code, maximum clarity.

Supported:
- Pinecone
- Chroma  
- Weaviate
- Qdrant
- Milvus
- PostgreSQL (pgvector)

Usage:
    from stonechain import Anthropic, RAG
    from stonechain_vectors import PineconeStore, ChromaStore
    
    # Use Pinecone instead of built-in similarity
    rag = RAG(Anthropic(), store=PineconeStore(api_key="...", index="docs"))
    
Author: Kent Stone
License: MIT
"""

__version__ = "1.0.0"

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import hashlib


# =============================================================================
# BASE INTERFACE
# =============================================================================

@dataclass
class VectorDocument:
    """Document with vector embedding."""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = None


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add(self, documents: List[VectorDocument]) -> None:
        """Add documents to the store."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[VectorDocument]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        pass
    
    def _generate_id(self, content: str) -> str:
        """Generate deterministic ID from content."""
        return hashlib.md5(content.encode()).hexdigest()[:16]


# =============================================================================
# PINECONE
# =============================================================================

class PineconeStore(VectorStore):
    """
    Pinecone vector database integration.
    
    Requires: pip install pinecone-client
    
    Usage:
        store = PineconeStore(
            api_key="your-api-key",
            index_name="my-index",
            environment="us-west1-gcp"  # or your region
        )
        rag = RAG(llm, store=store)
    """
    
    def __init__(
        self,
        api_key: str,
        index_name: str,
        environment: str = None,
        namespace: str = "",
        dimension: int = 1536  # OpenAI ada-002 default
    ):
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError("pip install pinecone-client")
        
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.namespace = namespace
        self.dimension = dimension
    
    def add(self, documents: List[VectorDocument]) -> None:
        """Upsert documents to Pinecone."""
        vectors = []
        for doc in documents:
            vectors.append({
                "id": doc.id or self._generate_id(doc.content),
                "values": doc.embedding,
                "metadata": {
                    "content": doc.content,
                    **(doc.metadata or {})
                }
            })
        
        # Batch upsert (Pinecone limit is 100)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[VectorDocument]:
        """Query Pinecone for similar vectors."""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace
        )
        
        docs = []
        for match in results.get("matches", []):
            docs.append(VectorDocument(
                id=match["id"],
                content=match.get("metadata", {}).get("content", ""),
                embedding=match.get("values", []),
                metadata=match.get("metadata", {})
            ))
        return docs
    
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID."""
        self.index.delete(ids=ids, namespace=self.namespace)


# =============================================================================
# CHROMA
# =============================================================================

class ChromaStore(VectorStore):
    """
    Chroma vector database integration.
    
    Requires: pip install chromadb
    
    Usage:
        # In-memory (default)
        store = ChromaStore(collection_name="my-docs")
        
        # Persistent
        store = ChromaStore(
            collection_name="my-docs",
            persist_directory="./chroma_db"
        )
        
        # Remote server
        store = ChromaStore(
            collection_name="my-docs",
            host="localhost",
            port=8000
        )
    """
    
    def __init__(
        self,
        collection_name: str = "stonechain",
        persist_directory: str = None,
        host: str = None,
        port: int = 8000
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError("pip install chromadb")
        
        if host:
            self.client = chromadb.HttpClient(host=host, port=port)
        elif persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add(self, documents: List[VectorDocument]) -> None:
        """Add documents to Chroma."""
        ids = []
        embeddings = []
        contents = []
        metadatas = []
        
        for doc in documents:
            ids.append(doc.id or self._generate_id(doc.content))
            embeddings.append(doc.embedding)
            contents.append(doc.content)
            metadatas.append(doc.metadata or {})
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[VectorDocument]:
        """Query Chroma for similar documents."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "embeddings"]
        )
        
        docs = []
        if results["ids"] and results["ids"][0]:
            for i, id in enumerate(results["ids"][0]):
                docs.append(VectorDocument(
                    id=id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    embedding=results["embeddings"][0][i] if results.get("embeddings") else [],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {}
                ))
        return docs
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        self.collection.delete(ids=ids)


# =============================================================================
# WEAVIATE
# =============================================================================

class WeaviateStore(VectorStore):
    """
    Weaviate vector database integration.
    
    Requires: pip install weaviate-client
    
    Usage:
        store = WeaviateStore(
            url="http://localhost:8080",
            class_name="Document"
        )
        
        # With API key (Weaviate Cloud)
        store = WeaviateStore(
            url="https://your-cluster.weaviate.network",
            api_key="your-api-key",
            class_name="Document"
        )
    """
    
    def __init__(
        self,
        url: str = "http://localhost:8080",
        api_key: str = None,
        class_name: str = "StoneChainDocument"
    ):
        try:
            import weaviate
        except ImportError:
            raise ImportError("pip install weaviate-client")
        
        auth = weaviate.auth.AuthApiKey(api_key) if api_key else None
        self.client = weaviate.Client(url=url, auth_client_secret=auth)
        self.class_name = class_name
        
        # Create schema if not exists
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create Weaviate class if it doesn't exist."""
        schema = {
            "class": self.class_name,
            "vectorizer": "none",  # We provide our own vectors
            "properties": [
                {"name": "content", "dataType": ["text"]},
                {"name": "metadata", "dataType": ["text"]}
            ]
        }
        
        if not self.client.schema.exists(self.class_name):
            self.client.schema.create_class(schema)
    
    def add(self, documents: List[VectorDocument]) -> None:
        """Add documents to Weaviate."""
        with self.client.batch as batch:
            for doc in documents:
                batch.add_data_object(
                    data_object={
                        "content": doc.content,
                        "metadata": json.dumps(doc.metadata or {})
                    },
                    class_name=self.class_name,
                    uuid=doc.id or self._generate_id(doc.content),
                    vector=doc.embedding
                )
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[VectorDocument]:
        """Query Weaviate for similar vectors."""
        result = (
            self.client.query
            .get(self.class_name, ["content", "metadata"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(top_k)
            .with_additional(["id", "vector"])
            .do()
        )
        
        docs = []
        data = result.get("data", {}).get("Get", {}).get(self.class_name, [])
        for item in data:
            docs.append(VectorDocument(
                id=item.get("_additional", {}).get("id", ""),
                content=item.get("content", ""),
                embedding=item.get("_additional", {}).get("vector", []),
                metadata=json.loads(item.get("metadata", "{}"))
            ))
        return docs
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        for id in ids:
            self.client.data_object.delete(uuid=id, class_name=self.class_name)


# =============================================================================
# QDRANT
# =============================================================================

class QdrantStore(VectorStore):
    """
    Qdrant vector database integration.
    
    Requires: pip install qdrant-client
    
    Usage:
        # Local
        store = QdrantStore(collection_name="my-docs")
        
        # Remote
        store = QdrantStore(
            url="http://localhost:6333",
            collection_name="my-docs"
        )
        
        # Qdrant Cloud
        store = QdrantStore(
            url="https://xyz.qdrant.io",
            api_key="your-api-key",
            collection_name="my-docs"
        )
    """
    
    def __init__(
        self,
        collection_name: str = "stonechain",
        url: str = None,
        api_key: str = None,
        dimension: int = 1536
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("pip install qdrant-client")
        
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(":memory:")
        
        self.collection_name = collection_name
        self.dimension = dimension
        
        # Create collection if not exists
        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )
    
    def add(self, documents: List[VectorDocument]) -> None:
        """Add documents to Qdrant."""
        from qdrant_client.models import PointStruct
        
        points = []
        for i, doc in enumerate(documents):
            points.append(PointStruct(
                id=doc.id or self._generate_id(doc.content),
                vector=doc.embedding,
                payload={
                    "content": doc.content,
                    **(doc.metadata or {})
                }
            ))
        
        self.client.upsert(collection_name=self.collection_name, points=points)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[VectorDocument]:
        """Query Qdrant for similar vectors."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_vectors=True
        )
        
        docs = []
        for hit in results:
            docs.append(VectorDocument(
                id=str(hit.id),
                content=hit.payload.get("content", ""),
                embedding=hit.vector or [],
                metadata={k: v for k, v in hit.payload.items() if k != "content"}
            ))
        return docs
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        from qdrant_client.models import PointIdsList
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids)
        )


# =============================================================================
# MILVUS
# =============================================================================

class MilvusStore(VectorStore):
    """
    Milvus vector database integration.
    
    Requires: pip install pymilvus
    
    Usage:
        store = MilvusStore(
            host="localhost",
            port=19530,
            collection_name="my-docs"
        )
    """
    
    def __init__(
        self,
        collection_name: str = "stonechain",
        host: str = "localhost",
        port: int = 19530,
        dimension: int = 1536
    ):
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        except ImportError:
            raise ImportError("pip install pymilvus")
        
        connections.connect(host=host, port=port)
        
        self.collection_name = collection_name
        self.dimension = dimension
        
        # Create collection if not exists
        if not utility.has_collection(collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ]
            schema = CollectionSchema(fields)
            self.collection = Collection(name=collection_name, schema=schema)
            
            # Create index
            index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
            self.collection.create_index(field_name="embedding", index_params=index_params)
        else:
            self.collection = Collection(collection_name)
        
        self.collection.load()
    
    def add(self, documents: List[VectorDocument]) -> None:
        """Add documents to Milvus."""
        ids = []
        contents = []
        metadatas = []
        embeddings = []
        
        for doc in documents:
            ids.append(doc.id or self._generate_id(doc.content))
            contents.append(doc.content[:65535])  # Milvus VARCHAR limit
            metadatas.append(json.dumps(doc.metadata or {})[:65535])
            embeddings.append(doc.embedding)
        
        self.collection.insert([ids, contents, metadatas, embeddings])
        self.collection.flush()
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[VectorDocument]:
        """Query Milvus for similar vectors."""
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["content", "metadata"]
        )
        
        docs = []
        for hits in results:
            for hit in hits:
                docs.append(VectorDocument(
                    id=hit.id,
                    content=hit.entity.get("content", ""),
                    embedding=[],  # Milvus doesn't return vectors by default
                    metadata=json.loads(hit.entity.get("metadata", "{}"))
                ))
        return docs
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        expr = f'id in {ids}'
        self.collection.delete(expr)


# =============================================================================
# PGVECTOR (PostgreSQL)
# =============================================================================

class PgVectorStore(VectorStore):
    """
    PostgreSQL pgvector integration.
    
    Requires: pip install psycopg2-binary pgvector
    
    Usage:
        store = PgVectorStore(
            connection_string="postgresql://user:pass@localhost/db",
            table_name="documents"
        )
    """
    
    def __init__(
        self,
        connection_string: str,
        table_name: str = "stonechain_documents",
        dimension: int = 1536
    ):
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
        except ImportError:
            raise ImportError("pip install psycopg2-binary pgvector")
        
        self.conn = psycopg2.connect(connection_string)
        self.table_name = table_name
        self.dimension = dimension
        
        register_vector(self.conn)
        
        # Create table if not exists
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id VARCHAR(64) PRIMARY KEY,
                    content TEXT,
                    metadata JSONB,
                    embedding vector({dimension})
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx 
                ON {table_name} USING ivfflat (embedding vector_cosine_ops)
            """)
            self.conn.commit()
    
    def add(self, documents: List[VectorDocument]) -> None:
        """Add documents to PostgreSQL."""
        with self.conn.cursor() as cur:
            for doc in documents:
                cur.execute(f"""
                    INSERT INTO {self.table_name} (id, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                """, (
                    doc.id or self._generate_id(doc.content),
                    doc.content,
                    json.dumps(doc.metadata or {}),
                    doc.embedding
                ))
            self.conn.commit()
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[VectorDocument]:
        """Query PostgreSQL for similar vectors."""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, content, metadata, embedding
                FROM {self.table_name}
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (query_embedding, top_k))
            
            docs = []
            for row in cur.fetchall():
                docs.append(VectorDocument(
                    id=row[0],
                    content=row[1],
                    metadata=row[2] or {},
                    embedding=list(row[3]) if row[3] else []
                ))
            return docs
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table_name} WHERE id = ANY(%s)", (ids,))
            self.conn.commit()
    
    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()


# =============================================================================
# EMBEDDING PROVIDERS
# =============================================================================

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass
    
    def embed_one(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


class OpenAIEmbeddings(EmbeddingProvider):
    """
    OpenAI embeddings (ada-002, text-embedding-3-small, etc.)
    
    Requires: OPENAI_API_KEY env var
    
    Usage:
        embeddings = OpenAIEmbeddings()  # Uses ada-002
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    """
    
    def __init__(self, model: str = "text-embedding-ada-002", api_key: str = None):
        import os
        import urllib.request
        
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via OpenAI API."""
        import urllib.request
        import json
        
        data = json.dumps({"input": texts, "model": self.model}).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )
        
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
        
        # Sort by index to maintain order
        sorted_data = sorted(result["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]


class CohereEmbeddings(EmbeddingProvider):
    """
    Cohere embeddings.
    
    Requires: COHERE_API_KEY env var
    """
    
    def __init__(self, model: str = "embed-english-v3.0", api_key: str = None):
        import os
        self.model = model
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY required")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via Cohere API."""
        import urllib.request
        import json
        
        data = json.dumps({
            "texts": texts,
            "model": self.model,
            "input_type": "search_document"
        }).encode()
        
        req = urllib.request.Request(
            "https://api.cohere.ai/v1/embed",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )
        
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
        
        return result["embeddings"]


class VoyageEmbeddings(EmbeddingProvider):
    """
    Voyage AI embeddings (great for code and retrieval).
    
    Requires: VOYAGE_API_KEY env var
    """
    
    def __init__(self, model: str = "voyage-2", api_key: str = None):
        import os
        self.model = model
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY required")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via Voyage API."""
        import urllib.request
        import json
        
        data = json.dumps({"input": texts, "model": self.model}).encode()
        req = urllib.request.Request(
            "https://api.voyageai.com/v1/embeddings",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )
        
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
        
        return [item["embedding"] for item in result["data"]]


# =============================================================================
# RAG INTEGRATION HELPER
# =============================================================================

class VectorRAG:
    """
    Enhanced RAG with vector database backend.
    
    Usage:
        from stonechain import Anthropic
        from stonechain_vectors import VectorRAG, ChromaStore, OpenAIEmbeddings
        
        rag = VectorRAG(
            llm=Anthropic(),
            store=ChromaStore(persist_directory="./db"),
            embeddings=OpenAIEmbeddings()
        )
        
        # Add documents
        rag.add(["Document 1 content", "Document 2 content"])
        
        # Query
        answer = rag.query("What is in document 1?")
    """
    
    def __init__(
        self,
        llm,
        store: VectorStore,
        embeddings: EmbeddingProvider,
        system_prompt: str = None
    ):
        self.llm = llm
        self.store = store
        self.embeddings = embeddings
        self.system_prompt = system_prompt or (
            "Answer the question based on the provided context. "
            "If the context doesn't contain relevant information, say so."
        )
    
    def add(self, texts: List[str], metadatas: List[Dict] = None) -> None:
        """Add texts to the vector store."""
        embeddings = self.embeddings.embed(texts)
        
        docs = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            docs.append(VectorDocument(
                id=None,  # Auto-generate
                content=text,
                embedding=embedding,
                metadata=metadatas[i] if metadatas else None
            ))
        
        self.store.add(docs)
    
    def query(self, question: str, top_k: int = 5) -> str:
        """Query the RAG system."""
        # Embed the question
        query_embedding = self.embeddings.embed_one(question)
        
        # Search for relevant documents
        docs = self.store.search(query_embedding, top_k=top_k)
        
        # Build context
        context = "\n\n".join([f"[{i+1}] {doc.content}" for i, doc in enumerate(docs)])
        
        # Generate answer
        prompt = f"""Context:
{context}

Question: {question}

Answer:"""
        
        return self.llm(prompt, system=self.system_prompt)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base
    "VectorStore",
    "VectorDocument",
    
    # Stores
    "PineconeStore",
    "ChromaStore", 
    "WeaviateStore",
    "QdrantStore",
    "MilvusStore",
    "PgVectorStore",
    
    # Embeddings
    "EmbeddingProvider",
    "OpenAIEmbeddings",
    "CohereEmbeddings",
    "VoyageEmbeddings",
    
    # RAG
    "VectorRAG",
]


if __name__ == "__main__":
    print("StoneChain Vector Database Integrations")
    print("=" * 45)
    print(f"Version: {__version__}")
    print("\nSupported Vector Databases:")
    print("  - Pinecone    (pip install pinecone-client)")
    print("  - Chroma      (pip install chromadb)")
    print("  - Weaviate    (pip install weaviate-client)")
    print("  - Qdrant      (pip install qdrant-client)")
    print("  - Milvus      (pip install pymilvus)")
    print("  - PostgreSQL  (pip install psycopg2-binary pgvector)")
    print("\nSupported Embedding Providers:")
    print("  - OpenAI      (OPENAI_API_KEY)")
    print("  - Cohere      (COHERE_API_KEY)")
    print("  - Voyage AI   (VOYAGE_API_KEY)")
    print("\nUsage:")
    print("  from stonechain_vectors import VectorRAG, ChromaStore, OpenAIEmbeddings")
    print("  rag = VectorRAG(llm, ChromaStore(), OpenAIEmbeddings())")

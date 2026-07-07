import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class VectorStore:
    """
    ChromaDB Vector Store for document chunks
    Uses local persistent storage (no cloud/API needed)
    """
    
    def __init__(self, persist_dir=None, collection_name=None):
        # Get project root
        project_root = Path(__file__).parent.parent.parent
        
        # Load from .env or use defaults - DEFINE THESE FIRST
        self.persist_dir = persist_dir or os.getenv(
            "CHROMA_PERSIST_DIR", 
            str(project_root / "project" / "vector_db_storage" / "chroma_db")
        )
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", 
            "docling-paddle"
        )
        
        # Create directory if it doesn't exist
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        # NOW we can print them
        print(f"📂 Using ChromaDB at: {self.persist_dir}")
        print(f"📦 Collection: {self.collection_name}")
        
        # Initialize ChromaDB client (LOCAL - no API key needed)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Docling + PaddleOCR document chunks"}
        )
        
        # Initialize embedding model
        print("🤖 Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Vector store ready!")
    
    def add_chunks(self, chunks, source_name: str):
        """
        Add document chunks to ChromaDB
        Args:
            chunks: List of chunk dictionaries
            source_name: Name of the source document
        """
        documents, metadatas, ids = [], [], []
        
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if not text:
                continue
            
            meta = chunk.get("metadata", {})
            
            # Create safe metadata (ChromaDB requires specific types)
            safe_meta = {
                "page": int(meta.get("page") or 0),
                "type": str(meta.get("type") or "unknown"),
                "source": str(source_name),
                "confidence": float(meta.get("confidence") or 0.0) if meta.get("confidence") else 0.0
            }
            
            # Add bbox if available
            if "bbox" in meta and meta["bbox"]:
                bbox = meta["bbox"]
                safe_meta["bbox_x_min"] = int(bbox.get("x_min", 0))
                safe_meta["bbox_y_min"] = int(bbox.get("y_min", 0))
                safe_meta["bbox_x_max"] = int(bbox.get("x_max", 0))
                safe_meta["bbox_y_max"] = int(bbox.get("y_max", 0))
            
            documents.append(text)
            metadatas.append(safe_meta)
            ids.append(f"{source_name}_chunk_{i}")
        
        if not documents:
            print("⚠️  No valid chunks to insert.")
            return
        
        # Add to ChromaDB (embeddings are generated automatically)
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Stored {len(documents)} chunks in ChromaDB collection '{self.collection_name}'")
    
    def search(self, query: str, n_results=5):
        """
        Search for similar documents
        Args:
            query: Search query
            n_results: Number of results to return
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    
    def get_stats(self):
        """Get collection statistics"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "persist_directory": self.persist_dir
        }
    
    def reset_collection(self):
        """⚠️ Delete all data in collection"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        print(f"🗑️  Collection '{self.collection_name}' reset!")
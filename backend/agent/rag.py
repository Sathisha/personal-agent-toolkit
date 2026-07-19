import os
import re
import httpx
import numpy as np
from typing import List, Dict, Any, Optional

class DocumentChunk:
    def __init__(self, doc_name: str, text: str, chunk_index: int, embedding: Optional[List[float]] = None):
        self.doc_name = doc_name
        self.text = text
        self.chunk_index = chunk_index
        self.embedding = embedding

class RagStore:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.documents: Dict[str, str] = {} # doc_name -> full_text

    def clear(self):
        self.chunks.clear()
        self.documents.clear()

    def add_document(self, doc_name: str, text: str, chunk_size: int = 800, overlap: int = 150):
        self.documents[doc_name] = text
        
        # Clean text and split into chunks
        # Simple sliding window chunker
        words = text.split()
        if not words:
            return
            
        # Let's chunk by character length to keep code formatting intact
        chunks_text = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            # Try to split at a space or newline if possible
            if end < len(text):
                # Look back up to 100 characters for a newline or space
                lookback = text[max(start, end - 100):end]
                split_idx = max(lookback.rfind('\n'), lookback.rfind(' '))
                if split_idx != -1:
                    end = max(start, end - 100) + split_idx
            
            chunk_txt = text[start:end].strip()
            if chunk_txt:
                chunks_text.append(chunk_txt)
            start = end - overlap
            if start >= len(text) or end >= len(text):
                break
                
        for i, chunk_txt in enumerate(chunks_text):
            self.chunks.append(DocumentChunk(
                doc_name=doc_name,
                text=chunk_txt,
                chunk_index=i
            ))

    async def compute_embeddings(self, api_type: str, api_url: str, model_name: str, api_key: str = ""):
        """Compute embeddings for all chunks that don't have them yet."""
        unembedded_chunks = [c for c in self.chunks if c.embedding is None]
        if not unembedded_chunks:
            return

        texts = [c.text for c in unembedded_chunks]
        
        try:
            embeddings = await self._fetch_embeddings(api_type, api_url, model_name, texts, api_key)
            if embeddings and len(embeddings) == len(unembedded_chunks):
                for chunk, emb in zip(unembedded_chunks, embeddings):
                    chunk.embedding = emb
        except Exception as e:
            print(f"Error computing embeddings: {e}. Falling back to text search.")

    async def _fetch_embeddings(self, api_type: str, api_url: str, model_name: str, texts: List[str], api_key: str = "") -> Optional[List[List[float]]]:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=30) as client:
            if api_type == "ollama":
                # Ollama Embeddings API
                # Ollama accepts a single prompt or an array of inputs depending on version. Let's do batch or individual
                # Newer Ollama versions support: POST /api/embed with {"model": model, "input": [...]}
                try:
                    payload = {"model": model_name, "input": texts}
                    response = await client.post(f"{api_url}/api/embed", json=payload, headers=headers)
                    if response.status_code == 200:
                        data = response.json()
                        return data.get("embeddings")
                except Exception:
                    # Fallback to older /api/embeddings for each chunk individually
                    embeddings = []
                    for text in texts:
                        response = await client.post(
                            f"{api_url}/api/embeddings", 
                            json={"model": model_name, "prompt": text},
                            headers=headers
                        )
                        if response.status_code == 200:
                            embeddings.append(response.json().get("embedding"))
                        else:
                            return None
                    return embeddings

            elif api_type == "openai" or api_type == "lmstudio":
                # OpenAI-compatible /v1/embeddings
                payload = {"model": model_name, "input": texts}
                response = await client.post(f"{api_url}/v1/embeddings", json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    return [item.get("embedding") for item in data.get("data", [])]
                    
        return None

    async def query(self, query_text: str, top_k: int = 3, api_type: str = "ollama", api_url: str = "", model_name: str = "", api_key: str = "") -> str:
        if not self.chunks:
            return "No documents indexed in the library."

        # Attempt to compute query embedding if we are using embeddings and chunks have them
        query_emb = None
        has_embeddings = any(c.embedding is not None for c in self.chunks)

        if has_embeddings and api_url and model_name:
            try:
                embs = await self._fetch_embeddings(api_type, api_url, model_name, [query_text], api_key)
                if embs:
                    query_emb = embs[0]
            except Exception as e:
                print(f"Failed to embed query: {e}. Falling back to text-based retrieval.")

        if query_emb is not None:
            # Vector-based Cosine Similarity Search
            scores = []
            q_vec = np.array(query_emb)
            q_norm = np.linalg.norm(q_vec)
            
            for chunk in self.chunks:
                if chunk.embedding is not None:
                    c_vec = np.array(chunk.embedding)
                    c_norm = np.linalg.norm(c_vec)
                    if q_norm > 0 and c_norm > 0:
                        sim = np.dot(q_vec, c_vec) / (q_norm * c_norm)
                    else:
                        sim = 0.0
                    scores.append((sim, chunk))
                else:
                    scores.append((0.0, chunk))
                    
            # Sort descending
            scores.sort(key=lambda x: x[0], reverse=True)
            top_results = scores[:top_k]
            
            formatted_results = []
            for score, chunk in top_results:
                formatted_results.append(
                    f"--- Source: {chunk.doc_name} (Similarity: {score:.3f}) ---\n{chunk.text}\n"
                )
            return "\n".join(formatted_results)
            
        else:
            # Fallback Text Search: simple TF-IDF / term frequency
            # Split query into words
            query_words = set(re.findall(r'\w+', query_text.lower()))
            if not query_words:
                # Return first few chunks
                top_results = self.chunks[:top_k]
                return "\n".join([f"--- Source: {c.doc_name} ---\n{c.text}\n" for c in top_results])

            scores = []
            for chunk in self.chunks:
                chunk_words = re.findall(r'\w+', chunk.text.lower())
                # Calculate term overlap score
                overlap = sum(1 for w in query_words if w in chunk_words)
                # Normalize by chunk length
                score = overlap / (len(chunk_words) + 1)
                scores.append((score, chunk))

            scores.sort(key=lambda x: x[0], reverse=True)
            top_results = scores[:top_k]
            
            formatted_results = []
            for score, chunk in top_results:
                formatted_results.append(
                    f"--- Source: {chunk.doc_name} (Text Match Score: {score:.3f}) ---\n{chunk.text}\n"
                )
            return "\n".join(formatted_results)

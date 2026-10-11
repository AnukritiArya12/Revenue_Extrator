import json
import os
import logging
try:
    import faiss
except ImportError:
    faiss = None
import numpy as np

logger = logging.getLogger("extractor")


class SemanticSearch:

    def __init__(
        self,
        chunks_path: str = "data/processed/chunks.jsonl",
        index_path: str = "data/processed/semantic.index",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.chunks_path = chunks_path
        self.index_path = index_path
        self.model_name = model_name

        self.model = None
        self.chunks = []
        self.index = None

        # Disable semantic model for fast BM25 testing
        self.model = None
        self.reload()

    def _init_model(self):
        """Attempts to load SentenceTransformer model with local fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except Exception as e1:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception as e2:
                logger.warning(
                    "Could not load SentenceTransformer model (%s). Operating in offline mode (BM25 fallback).",
                    e1,
                )
                self.model = None

    def reload(self):
        """Reloads processed text chunks from disk and synchronizes the FAISS index."""
        self.chunks = self._load_chunks()
        if self.model is not None:
            self._load_or_build_index()

    def _load_chunks(self) -> list:
        """Reads and parses chunk dictionaries from the processed JSONL file."""
        if not os.path.exists(self.chunks_path):
            return []

        chunks = []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        chunks.append(json.loads(line_str))
                    except json.JSONDecodeError:
                        continue
        return chunks

    def _build_index(self):
        """Generates normalized vector embeddings and saves a new FAISS index file."""
        if not self.chunks or self.model is None:
            self.index = None
            return

        try:
            import faiss
        except ImportError:
            logger.warning("faiss is not installed. Operating in offline BM25 fallback mode.")
            self.index = None
            return

        texts = [chunk.get("text", "") for chunk in self.chunks]

        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            embeddings = np.asarray(embeddings, dtype="float32")
            dimension = embeddings.shape[1]

            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)

            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
        except Exception as err:
            logger.warning("Error building FAISS index: %s", err)
            self.index = None

    def _load_or_build_index(self):
        """Loads FAISS index from disk or rebuilds if missing, invalid, or out-of-sync."""
        if not self.chunks or self.model is None:
            self.index = None
            return

        try:
            import faiss
        except ImportError:
            self.index = None
            return

        if os.path.exists(self.index_path):
            try:
                loaded_index = faiss.read_index(self.index_path)
                if loaded_index.ntotal == len(self.chunks):
                    self.index = loaded_index
                    return
            except Exception as err:
                logger.warning("Could not read FAISS index (%s). Rebuilding...", err)

        self._build_index()

    def search(self, query: str, top_k: int = 10) -> list:
        """
        Executes similarity search for a query and returns chunks with semantic scores.
        Falls back safely to an empty list if offline or model is unavailable.
        """
        if not self.chunks or self.model is None:
            return []

        if self.index is None or self.index.ntotal != len(self.chunks):
            self._load_or_build_index()

        if self.index is None:
            return []

        try:
            query_embedding = self.model.encode(
                [query],
                normalize_embeddings=True,
            )
            query_embedding = np.asarray(query_embedding, dtype="float32")

            k = min(top_k, len(self.chunks))
            if k <= 0:
                return []

            scores, indices = self.index.search(query_embedding, k)

            results = []
            for score, index in zip(scores[0], indices[0]):
                if index < 0 or index >= len(self.chunks):
                    continue

                chunk = self.chunks[index].copy()
                chunk["semantic_score"] = float(score)
                results.append(chunk)

            return results
        except Exception as err:
            logger.warning("Semantic search failed: %s. Reverting to BM25.", err)
            return []
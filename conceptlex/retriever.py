import json
from pathlib import Path
import numpy as np

class Retriever:

    def __init__(self, corpus_dir: str | Path):
        self.corpus_dir = Path(corpus_dir)
        self._model = None
        self.indexes: dict[str, dict] = {}
        for jur_dir in self.corpus_dir.iterdir():
            if not jur_dir.is_dir():
                continue
            embeddings_path = jur_dir / 'embeddings.npy'
            chunks_path = jur_dir / 'chunks.jsonl'
            if not (embeddings_path.exists() and chunks_path.exists()):
                continue
            self.indexes[jur_dir.name] = {'embeddings': np.load(embeddings_path), 'chunks': [json.loads(l) for l in chunks_path.read_text().splitlines() if l.strip()]}

    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        return self._model

    def embed(self, queries: list[str]) -> np.ndarray:
        result = self.model.encode(queries, batch_size=64, max_length=512, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return np.asarray(result['dense_vecs'], dtype=np.float32)

    def top_k(self, query: str, jurisdiction: str, k: int=3) -> list[str]:
        if jurisdiction not in self.indexes:
            return []
        q_emb = self.embed([query])[0]
        emb_matrix = self.indexes[jurisdiction]['embeddings']
        sims = emb_matrix @ q_emb
        if k >= len(sims):
            order = np.argsort(-sims)
        else:
            top = np.argpartition(-sims, k)[:k]
            order = top[np.argsort(-sims[top])]
        chunks = self.indexes[jurisdiction]['chunks']
        return [chunks[i]['text'] for i in order[:k]]

    def top_k_plausibility_matched(self, query: str, target_jur: str, other_jur: str, k: int=3) -> list[str]:
        if target_jur not in self.indexes or other_jur not in self.indexes:
            return []
        q = self.embed([query])[0]
        tgt_sims = self.indexes[target_jur]['embeddings'] @ q
        tgt_profile = np.sort(tgt_sims)[::-1][:k]
        oth_sims = self.indexes[other_jur]['embeddings'] @ q
        oth_chunks = self.indexes[other_jur]['chunks']
        chosen, used = ([], set())
        for s in tgt_profile:
            for idx in np.argsort(np.abs(oth_sims - s)):
                idx = int(idx)
                if idx not in used:
                    used.add(idx)
                    chosen.append(oth_chunks[idx]['text'])
                    break
        return chosen

    def top_k_batch(self, queries: list[str], jurisdiction: str, k: int=3) -> list[list[str]]:
        if jurisdiction not in self.indexes:
            return [[] for _ in queries]
        q_emb = self.embed(queries)
        emb_matrix = self.indexes[jurisdiction]['embeddings']
        sims = q_emb @ emb_matrix.T
        chunks = self.indexes[jurisdiction]['chunks']
        results = []
        for row in sims:
            if k >= len(row):
                order = np.argsort(-row)
            else:
                top = np.argpartition(-row, k)[:k]
                order = top[np.argsort(-row[top])]
            results.append([chunks[i]['text'] for i in order[:k]])
        return results

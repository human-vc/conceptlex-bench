import io
import json
import re
import zipfile
from pathlib import Path
import numpy as np
import requests
from datasets import load_dataset
CHUNK_TOKEN_TARGET = 512
CHUNK_OVERLAP_TOKENS = 64
JURISDICTIONS = [{'code': 'US', 'language': 'en'}, {'code': 'DE', 'language': 'de'}, {'code': 'FR', 'language': 'fr'}, {'code': 'CN', 'language': 'en'}]
HKSAR_ZIP_URLS = ['https://www.elegislation.gov.hk/data/full/hk-doj-hkel-legislation-current-en-caps-1-300.zip', 'https://www.elegislation.gov.hk/data/full/hk-doj-hkel-legislation-current-en-caps-301-600.zip', 'https://www.elegislation.gov.hk/data/full/hk-doj-hkel-legislation-current-en-caps-601-end.zip']

def _take(stream, n):
    out = []
    for row in stream:
        out.append(row)
        if len(out) >= n:
            break
    return out

def load_us(target_n: int=30000) -> list[dict]:
    docs = []
    try:
        cases = load_dataset('common-pile/caselaw_access_project', split='train', streaming=True)
        for r in _take(cases, target_n):
            metadata = r.get('metadata') or {}
            docs.append({'text': r.get('text', ''), 'source_url': metadata.get('url', ''), 'instrument_type': 'case', 'year': (r.get('created') or '')[:4] or None, 'doc_id': r.get('id', '')})
    except Exception as e:
        print(f'US case-law load failed: {e}')
    return docs

def load_de(target_n: int=30000) -> list[dict]:
    return []

def load_fr(target_n: int=30000) -> list[dict]:
    docs = []
    try:
        ds = load_dataset('AgentPublic/legi', split='train', streaming=True)
        for r in ds:
            if r.get('status') != 'VIGUEUR':
                continue
            emb_field = r.get('embeddings_bge-m3')
            try:
                emb = json.loads(emb_field) if isinstance(emb_field, str) else emb_field
            except (json.JSONDecodeError, TypeError):
                emb = None
            docs.append({'_pre_chunked': True, 'text': r.get('chunk_text', ''), 'embedding': emb, 'source_url': f"https://www.legifrance.gouv.fr/{r.get('doc_id', '')}", 'instrument_type': (r.get('category') or '').lower() or 'statute', 'year': (r.get('start_date') or '')[:4] or None, 'doc_id': r.get('doc_id', ''), 'chunk_idx': r.get('chunk_index', 0)})
            if len(docs) >= target_n:
                break
    except Exception as e:
        print(f'FR AgentPublic/legi load failed: {e}')
    return docs

def load_cn(target_n: int=30000) -> list[dict]:
    docs = []
    for url in HKSAR_ZIP_URLS:
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            for name in zf.namelist():
                if not name.lower().endswith('.xml'):
                    continue
                try:
                    raw = zf.read(name).decode('utf-8', errors='ignore')
                except Exception:
                    continue
                text = re.sub('<[^>]+>', ' ', raw)
                text = re.sub('\\s+', ' ', text).strip()
                if len(text) < 200:
                    continue
                docs.append({'text': text, 'source_url': f'https://www.elegislation.gov.hk/{name}', 'instrument_type': 'statute', 'year': None, 'doc_id': name})
                if len(docs) >= target_n:
                    return docs
        except Exception as e:
            print(f'HKSAR fetch failed for {url}: {e}')
    return docs

def chunk_with_metadata(documents: list[dict], jurisdiction: str, language: str) -> list[dict]:
    from transformers import AutoTokenizer
    import semchunk
    tok = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    chunker = semchunk.chunkerify(tok, chunk_size=CHUNK_TOKEN_TARGET)
    chunks = []
    for d in documents:
        if d.get('_pre_chunked'):
            chunks.append({'chunk_id': f"{jurisdiction}/{d['doc_id']}/{d['chunk_idx']:04d}", 'doc_id': d['doc_id'], 'chunk_idx': d['chunk_idx'], 'jurisdiction': jurisdiction, 'language': language, 'instrument_type': d['instrument_type'], 'source_url': d['source_url'], 'year': d.get('year'), 'text': d['text'], '_pre_embedded': True, '_embedding': d.get('embedding')})
            continue
        text = d.get('text') or ''
        if not text.strip():
            continue
        pieces = chunker(text)
        if isinstance(pieces, tuple):
            pieces = pieces[0]
        doc_id = re.sub('[^a-zA-Z0-9_.-]+', '_', d.get('doc_id', '') or 'doc')[:200]
        for i, piece in enumerate(pieces):
            chunks.append({'chunk_id': f'{jurisdiction}/{doc_id}/{i:04d}', 'doc_id': doc_id, 'chunk_idx': i, 'jurisdiction': jurisdiction, 'language': language, 'instrument_type': d.get('instrument_type'), 'source_url': d.get('source_url'), 'year': d.get('year'), 'text': piece, '_pre_embedded': False})
    return chunks

def embed_chunks_bge_m3(chunks: list[dict], batch_size: int=128) -> np.ndarray:
    from FlagEmbedding import BGEM3FlagModel
    needs_embed = [c for c in chunks if not c.get('_pre_embedded') or c.get('_embedding') is None]
    if needs_embed:
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        texts = [c['text'] for c in needs_embed]
        result = model.encode(texts, batch_size=batch_size, max_length=CHUNK_TOKEN_TARGET, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        for c, v in zip(needs_embed, result['dense_vecs']):
            c['_embedding'] = np.asarray(v, dtype=np.float32).tolist()
    return np.array([c['_embedding'] for c in chunks], dtype=np.float32)

def save_per_jurisdiction(chunks: list[dict], embeddings: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'embeddings.npy', embeddings)
    with (out_dir / 'chunks.jsonl').open('w') as f:
        for c in chunks:
            stripped = {k: v for k, v in c.items() if not k.startswith('_')}
            f.write(json.dumps(stripped, ensure_ascii=False) + '\n')

def build_corpus(output_dir: Path, target_per_jurisdiction: int=30000):
    loaders = {'US': load_us, 'DE': load_de, 'FR': load_fr, 'CN': load_cn}
    language_map = {'US': 'en', 'DE': 'de', 'FR': 'fr', 'CN': 'en'}
    output_dir.mkdir(parents=True, exist_ok=True)
    for jur_code, loader in loaders.items():
        print(f'\n=== Loading {jur_code} ===')
        docs = loader(target_per_jurisdiction)
        print(f'{jur_code}: loaded {len(docs)} documents')
        if not docs:
            continue
        chunks = chunk_with_metadata(docs, jur_code, language_map[jur_code])
        print(f'{jur_code}: chunked into {len(chunks)} chunks')
        if not chunks:
            continue
        embeddings = embed_chunks_bge_m3(chunks)
        print(f'{jur_code}: embedded {embeddings.shape}')
        save_per_jurisdiction(chunks, embeddings, output_dir / jur_code)
        print(f'{jur_code}: saved to {output_dir / jur_code}')

def main():
    import sys
    if len(sys.argv) < 2:
        print('usage: python -m conceptlex.build_corpora <output_dir> [target_per_jurisdiction]')
        sys.exit(1)
    output_dir = Path(sys.argv[1])
    target_n = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    build_corpus(output_dir, target_n)
if __name__ == '__main__':
    main()

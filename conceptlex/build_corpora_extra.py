import json
import re
import subprocess
import time
from pathlib import Path
import numpy as np
import requests
from conceptlex.build_corpora import chunk_with_metadata, embed_chunks_bge_m3, save_per_jurisdiction
QUANTLAW_REPO = 'https://github.com/QuantLaw/gesetze-im-internet.git'
NPC_LAW_INDEX_PAGES = ['http://www.npc.gov.cn/englishnpc/c2759/common_list.shtml', 'http://www.npc.gov.cn/englishnpc/c2759/common_list_2.shtml']
OLD_DE_API = 'https://de.openlegaldata.io/api/cases/?format=json&limit=100'
HKSAR_API_BASE = 'https://www.elegislation.gov.hk/api/exportContent'
HKSAR_LISTING_API = 'https://www.elegislation.gov.hk/api/searchLegislation?lang=en'

def _clone_quantlaw_into(tmp_dir: Path) -> Path:
    target = tmp_dir / 'gesetze-im-internet'
    if target.exists():
        return target
    subprocess.check_call(['git', 'clone', '--depth', '1', '--branch', 'data', '--single-branch', QUANTLAW_REPO, str(target)])
    return target

def _parse_gii_xml(xml_text: str) -> str:
    text = re.sub('<[^>]+>', ' ', xml_text)
    text = re.sub('\\s+', ' ', text).strip()
    return text

def load_de_quantlaw(target_n: int=30000) -> list[dict]:
    tmp = Path('/tmp/conceptlex_corpora_src')
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        repo = _clone_quantlaw_into(tmp)
    except subprocess.CalledProcessError as e:
        print(f'QuantLaw clone failed: {e}')
        return []
    docs = []
    for xml_path in sorted(repo.rglob('*.xml')):
        try:
            raw = xml_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        text = _parse_gii_xml(raw)
        if len(text) < 500:
            continue
        rel = xml_path.relative_to(repo)
        docs.append({'text': text, 'source_url': f'https://www.gesetze-im-internet.de/{rel}', 'instrument_type': 'statute', 'year': None, 'doc_id': str(rel)})
        if len(docs) >= target_n:
            break
    return docs

def load_de_open_legal_data(target_n: int=30000) -> list[dict]:
    docs = []
    url = OLD_DE_API
    while url and len(docs) < target_n:
        try:
            r = requests.get(url, timeout=120, headers={'User-Agent': 'Mozilla/5.0'})
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f'  openlegaldata.io page failed: {e}')
            break
        for c in data.get('results', []):
            text = (c.get('content') or c.get('text') or '').strip()
            if len(text) < 500:
                continue
            docs.append({'text': text, 'source_url': c.get('source') or c.get('url') or '', 'instrument_type': 'case', 'year': str(c.get('date', ''))[:4] or None, 'doc_id': str(c.get('id', ''))})
            if len(docs) >= target_n:
                break
        url = data.get('next')
        time.sleep(0.3)
    return docs

def load_de(target_n: int=30000) -> list[dict]:
    print('  trying QuantLaw gesetze-im-internet mirror...')
    docs = load_de_quantlaw(target_n)
    if docs:
        print(f'  QuantLaw -> {len(docs)} statutes')
        return docs
    print('  trying open-legal-data...')
    docs = load_de_open_legal_data(target_n)
    if docs:
        print(f'  open-legal-data -> {len(docs)} cases')
        return docs
    return []

def _hksar_listing() -> list[dict]:
    try:
        r = requests.get(HKSAR_LISTING_API, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get('results', []) or data.get('legislations', []) or []
    except Exception as e:
        print(f'  HKSAR listing fetch failed: {e}')
        return []

def _hksar_fetch_text(cap_no: str, lang: str='en') -> str:
    try:
        params = {'cap': cap_no, 'lang': lang, 'format': 'html'}
        r = requests.get(HKSAR_API_BASE, params=params, timeout=120)
        if r.status_code != 200:
            return ''
        return _parse_gii_xml(r.text)
    except Exception:
        return ''

def load_cn_hksar(target_n: int=30000) -> list[dict]:
    return []

def load_cn_npc(target_n: int=30000) -> list[dict]:
    docs = []
    seen_urls = set()
    for index_url in NPC_LAW_INDEX_PAGES:
        try:
            r = requests.get(index_url, timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code != 200:
                continue
            html = r.text
        except Exception:
            continue
        for m in re.finditer('href="([^"]+\\.shtml)"[^>]*>([^<]+)</a>', html):
            link = m.group(1)
            if not link.startswith('http'):
                link = 'http://en.npc.gov.cn.cdurl.cn' + (link if link.startswith('/') else '/' + link)
            if link in seen_urls:
                continue
            seen_urls.add(link)
            try:
                page = requests.get(link, timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
                if page.status_code != 200:
                    continue
                text = _parse_gii_xml(page.text)
                if len(text) < 500:
                    continue
                docs.append({'text': text, 'source_url': link, 'instrument_type': 'statute', 'year': None, 'doc_id': link.rsplit('/', 1)[-1].replace('.shtml', '')})
                if len(docs) >= target_n:
                    return docs
                time.sleep(0.3)
            except Exception:
                continue
    return docs

def load_cn(target_n: int=30000) -> list[dict]:
    print('  trying HKSAR e-Legislation API...')
    docs = load_cn_hksar(target_n)
    if docs:
        print(f'  HKSAR API -> {len(docs)} statutes')
        return docs
    print('  trying NPC bilingual scrape...')
    docs = load_cn_npc(target_n)
    if docs:
        print(f'  NPC scrape -> {len(docs)} statutes')
        return docs
    return []

def build_missing(output_dir: Path, target_per_jurisdiction: int=15000):
    jurisdictions = [('DE', 'de', load_de), ('CN', 'en', load_cn)]
    output_dir.mkdir(parents=True, exist_ok=True)
    for jur_code, language, loader in jurisdictions:
        existing = output_dir / jur_code / 'embeddings.npy'
        if existing.exists():
            print(f'=== {jur_code} already built; skipping ===')
            continue
        print(f'\n=== Loading {jur_code} ===')
        docs = loader(target_per_jurisdiction)
        print(f'{jur_code}: loaded {len(docs)} documents')
        if not docs:
            print(f'{jur_code}: skipping (no documents)')
            continue
        chunks = chunk_with_metadata(docs, jur_code, language)
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
        print('usage: python -m conceptlex.build_corpora_extra <output_dir> [target_per_jurisdiction]')
        sys.exit(1)
    output_dir = Path(sys.argv[1])
    target_n = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
    build_missing(output_dir, target_n)
if __name__ == '__main__':
    main()

import hashlib
import json
import random
from pathlib import Path
from typing import Literal
DistractorType = Literal['lexical_false_friend', 'doctrinal_sibling', 'plausible_misapplication']
LANG_QUESTION_TEMPLATES = {'en': 'Under the law of {jurisdiction}, what is the correct doctrinal treatment of {term}?', 'de': 'Wie wird {term} im Recht von {jurisdiction} doktrinell behandelt?', 'fr': 'Selon le droit de {jurisdiction}, quel est le traitement doctrinal correct de {term}?', 'zh': '根据{jurisdiction}的法律,{term}的正确教义处理是什么?'}
JURISDICTION_NAMES = {'en': {'US': 'the United States', 'DE': 'Germany', 'FR': 'France', 'CN': 'China', 'UK': 'the United Kingdom', 'HK': 'Hong Kong'}, 'de': {'US': 'den USA', 'DE': 'Deutschland', 'FR': 'Frankreich', 'CN': 'China', 'UK': 'dem Vereinigten Königreich', 'HK': 'Hongkong'}, 'fr': {'US': 'les États-Unis', 'DE': "l'Allemagne", 'FR': 'la France', 'CN': 'la Chine', 'UK': 'le Royaume-Uni', 'HK': 'Hong Kong'}, 'zh': {'US': '美国', 'DE': '德国', 'FR': '法国', 'CN': '中国', 'UK': '英国', 'HK': '香港'}}

def stable_seed(pair_id: str, direction: str) -> int:
    digest = hashlib.sha256(f'{pair_id}::{direction}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'big') % 2 ** 31

def build_question(concept: dict, jurisdiction: str, lang: str) -> str:
    juris_name = JURISDICTION_NAMES.get(lang, JURISDICTION_NAMES['en']).get(jurisdiction, jurisdiction)
    return LANG_QUESTION_TEMPLATES[lang].format(term=concept['term'], jurisdiction=juris_name)

def correct_answer(target_concept: dict) -> str:
    anchor = target_concept.get('statutory_anchor', '')
    anchor_str = f' ({anchor})' if anchor else ''
    return f"{target_concept['definition_short']}{anchor_str}"

def build_distractors(pair: dict, direction: str) -> list[tuple[str, str]]:
    target_key = direction.split('_to_')[1]
    distractor_pool = pair.get('distractors', {}).get(direction, [])
    if not distractor_pool:
        source_key = direction.split('_to_')[0]
        source_concept = pair[f'concept_{source_key}']
        target_concept = pair[f'concept_{target_key}']
        distractor_pool = [{'type': 'lexical_false_friend', 'text': source_concept['definition_short']}, {'type': 'doctrinal_sibling', 'text': f"The doctrine is treated as a sub-category of the broader notion of {target_concept['term']} without further jurisdiction-specific qualification."}, {'type': 'plausible_misapplication', 'text': f"The doctrine applies only in the procedural context of {target_concept.get('doctrinal_area', 'contracts')} and does not extend to substantive analysis."}]
    return [(d['type'], d['text']) for d in distractor_pool[:3]]

def make_mcq_item(pair: dict, direction: str) -> dict:
    target_key = direction.split('_to_')[1]
    target_concept = pair[f'concept_{target_key}']
    target_jur = pair[f'jurisdiction_{target_key}']
    lang = target_concept['language']
    distractors = build_distractors(pair, direction)
    distractor_types = [t for t, _ in distractors]
    options = [correct_answer(target_concept)] + [text for _, text in distractors]
    rng = random.Random(stable_seed(pair['id'], f'mcq-{direction}'))
    order = list(range(len(options)))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    correct_idx = order.index(0)
    shuffled_distractor_types = [distractor_types[order[i] - 1] if order[i] > 0 else 'correct' for i in range(len(order))]
    return {'id': f"mcq-{pair['id'][3:]}-{direction}", 'concept_pair_id': pair['id'], 'direction': direction, 'language': lang, 'question': build_question(target_concept, target_jur, lang), 'options': shuffled, 'correct_idx': correct_idx, 'option_types': shuffled_distractor_types, 'expected_frame': f'pure_{target_key}'}

def make_generative_probe(pair: dict, direction: str) -> dict:
    target_key = direction.split('_to_')[1]
    target_concept = pair[f'concept_{target_key}']
    target_jur = pair[f'jurisdiction_{target_key}']
    lang = target_concept['language']
    fact_patterns = pair.get('fact_patterns', {})
    fact_pattern = fact_patterns.get(direction)
    if not fact_pattern:
        juris_name = JURISDICTION_NAMES.get(lang, JURISDICTION_NAMES['en']).get(target_jur, target_jur)
        fact_pattern = f"Analyze the following situation under {juris_name} law in the context of {target_concept['term']}. Parties A and B are engaged in a dispute that requires application of the controlling doctrine. Identify the applicable rule, apply it to the facts, and state the conclusion under {juris_name} law."
    rubric_keys = [k for k in [target_concept.get('statutory_anchor'), target_concept.get('term'), target_jur] if k]
    return {'id': f"gp-{pair['id'][3:]}-{direction}", 'concept_pair_id': pair['id'], 'fact_pattern': fact_pattern, 'language': lang, 'target_jurisdiction': target_jur, 'expected_frame': f'pure_{target_key}', 'rubric_keys': rubric_keys}

def make_retrieval_contamination_item(pair: dict, direction: str) -> dict:
    target_key = direction.split('_to_')[1]
    source_key = direction.split('_to_')[0]
    target_concept = pair[f'concept_{target_key}']
    target_jur = pair[f'jurisdiction_{target_key}']
    other_jur = pair[f'jurisdiction_{source_key}']
    lang = target_concept['language']
    return {'id': f"rci-{pair['id'][3:]}-{direction}", 'concept_pair_id': pair['id'], 'question': build_question(target_concept, target_jur, lang), 'language': lang, 'target_jurisdiction': target_jur, 'retrieval_conditions': [{'condition': 'match', 'corpus_jurisdiction': target_jur, 'expected_frame_if_grounded': f'pure_{target_key}'}, {'condition': 'mismatch', 'corpus_jurisdiction': other_jur, 'expected_frame_if_grounded': 'frame_shift'}, {'condition': 'empty', 'corpus_jurisdiction': None, 'expected_frame_if_grounded': f'pure_{target_key}'}]}

def build_benchmark(pairs_path: Path, output_dir: Path) -> dict:
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    mcq, generative, retrieval = ([], [], [])
    for pair in pairs:
        for direction in ('a_to_b', 'b_to_a'):
            mcq.append(make_mcq_item(pair, direction))
            generative.append(make_generative_probe(pair, direction))
            retrieval.append(make_retrieval_contamination_item(pair, direction))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'mcq.jsonl').write_text('\n'.join((json.dumps(x, ensure_ascii=False) for x in mcq)))
    (output_dir / 'generative.jsonl').write_text('\n'.join((json.dumps(x, ensure_ascii=False) for x in generative)))
    (output_dir / 'retrieval_contamination.jsonl').write_text('\n'.join((json.dumps(x, ensure_ascii=False) for x in retrieval)))
    return {'mcq': len(mcq), 'generative': len(generative), 'retrieval_contamination': len(retrieval)}

def main():
    import sys
    if len(sys.argv) < 3:
        print('usage: python -m conceptlex.generate_items <pairs.jsonl> <output_dir>')
        sys.exit(1)
    stats = build_benchmark(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(stats, indent=2))
if __name__ == '__main__':
    main()

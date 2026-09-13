import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from conceptlex.judge_prompts import FEW_SHOTS, LABELS, LABEL_DEFS, SYSTEM
JUDGE_MODEL = os.environ.get('CONCEPTLEX_JUDGE_MODEL', 'ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4')
JSON_RE = re.compile('<<<JSON>>>(.*?)<<<END>>>', re.DOTALL)
LABEL_RE = re.compile('"?label"?\\s*:\\s*"?(' + '|'.join(LABELS) + ')"?', re.IGNORECASE)
CONF_RE = re.compile('"?confidence"?\\s*:\\s*"?([1-5])"?')
REFUSAL_RE = re.compile("^\\s*(I can't|I cannot|I'm not able|As an AI|I am unable)", re.I)
PAIRS_URL_LOCAL = Path(__file__).resolve().parent.parent / 'data' / 'concept_pairs.jsonl'

def load_concept_pairs() -> dict[str, dict]:
    pairs = {}
    if PAIRS_URL_LOCAL.exists():
        for line in PAIRS_URL_LOCAL.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                pairs[d['id']] = d
    return pairs

def pair_id_from_item(item_id: str) -> str:
    parts = item_id.split('-')
    if len(parts) >= 3:
        return 'cp-' + '-'.join(parts[1:-1])
    return ''

def direction_from_item(item_id: str) -> str:
    if item_id.endswith('a_to_b'):
        return 'a_to_b'
    if item_id.endswith('b_to_a'):
        return 'b_to_a'
    return ''

def enrich_response(response: dict, pairs: dict[str, dict]) -> dict:
    pair_id = pair_id_from_item(response.get('item_id', ''))
    direction = direction_from_item(response.get('item_id', ''))
    pair = pairs.get(pair_id, {})
    if direction == 'a_to_b':
        target_jur = pair.get('jurisdiction_b', '?')
        other_jur = pair.get('jurisdiction_a', '?')
        target_concept = pair.get('concept_b', {}).get('term', '?')
        other_concept = pair.get('concept_a', {}).get('term', '?')
    elif direction == 'b_to_a':
        target_jur = pair.get('jurisdiction_a', '?')
        other_jur = pair.get('jurisdiction_b', '?')
        target_concept = pair.get('concept_a', {}).get('term', '?')
        other_concept = pair.get('concept_b', {}).get('term', '?')
    else:
        target_jur = other_jur = target_concept = other_concept = '?'
    return {**response, 'target_jurisdiction': target_jur, 'other_jurisdiction': other_jur, 'target_concept': target_concept, 'other_concept': other_concept, 'concept_pair_id': pair_id, 'direction': direction}

def build_judge_messages(rec: dict) -> list[dict]:
    fs_blocks = []
    for ex in FEW_SHOTS:
        fs_blocks.append(f'''QUESTION: {ex['question']}\nTARGET: {ex['target_jurisdiction']} (concept: {ex['target_concept']})\nOTHER: {ex['other_jurisdiction']} (concept: {ex['other_concept']})\nRESPONSE: {ex['response']}\n<<<JSON>>>{{"reasoning":{json.dumps(ex['reasoning'])},"label":"{ex['label']}","confidence":{ex['confidence']}}}<<<END>>>''')
    question_or_prompt = rec.get('question') or rec.get('fact_pattern') or ''
    user = LABEL_DEFS + '\n\nEXAMPLES:\n\n' + '\n\n'.join(fs_blocks) + '\n\nNOW LABEL THIS ITEM:\n' + f'QUESTION: {question_or_prompt}\n' + f"TARGET: {rec['target_jurisdiction']} (concept: {rec['target_concept']})\n" + f"OTHER: {rec['other_jurisdiction']} (concept: {rec['other_concept']})\n" + f"RESPONSE: {rec.get('response_text', '')}\n"
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]

def parse_judge_output(out: str) -> tuple[str, Optional[int], str]:
    if not out:
        return ('__PARSE_FAIL__', None, '')
    if REFUSAL_RE.search(out.strip()):
        return ('__JUDGE_REFUSED__', None, out)
    m = JSON_RE.search(out)
    if m:
        candidate = m.group(1).strip()
        try:
            obj = json.loads(candidate)
            label = (obj.get('label') or '').strip().lower()
            if label in LABELS:
                conf = obj.get('confidence')
                try:
                    conf = int(conf) if conf is not None else None
                except (TypeError, ValueError):
                    conf = None
                return (label, conf, obj.get('reasoning') or '')
        except json.JSONDecodeError:
            pass
    m2 = LABEL_RE.search(out)
    if m2:
        cm = CONF_RE.search(out)
        return (m2.group(1).lower(), int(cm.group(1)) if cm else None, out)
    return ('__PARSE_FAIL__', None, out)

def silver_label_responses(responses_path: Path, pairs: dict[str, dict], output_path: Path):
    from vllm import LLM, SamplingParams
    records = [json.loads(l) for l in responses_path.read_text().splitlines() if l.strip()]
    enriched = [enrich_response(r, pairs) for r in records]
    is_awq = 'AWQ' in JUDGE_MODEL
    llm = LLM(model=JUDGE_MODEL, dtype='float16' if is_awq else 'bfloat16', max_model_len=8192, gpu_memory_utilization=0.92, max_num_seqs=16, max_num_batched_tokens=8192, enable_prefix_caching=True, trust_remote_code=True, seed=42)
    tokenizer = llm.get_tokenizer()
    tmpl_kwargs = {'tokenize': False, 'add_generation_prompt': True}
    if 'qwen3' in JUDGE_MODEL.lower():
        tmpl_kwargs['enable_thinking'] = False

    def render(rec):
        try:
            return tokenizer.apply_chat_template(build_judge_messages(rec), **tmpl_kwargs)
        except TypeError:
            tmpl_kwargs.pop('enable_thinking', None)
            return tokenizer.apply_chat_template(build_judge_messages(rec), **tmpl_kwargs)
    prompts = [render(rec) for rec in enriched]
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=240, seed=42, stop=['<<<END>>>'])
    outputs = llm.generate(prompts, sampling)
    with output_path.open('w') as f:
        for rec, out in zip(enriched, outputs):
            text = out.outputs[0].text + '<<<END>>>'
            label, conf, rationale = parse_judge_output(text)
            silver = {'response_id': rec.get('id'), 'item_id': rec.get('item_id'), 'model': rec.get('model'), 'item_type': rec.get('item_type'), 'language': rec.get('language') or 'en', 'concept_pair_id': rec.get('concept_pair_id'), 'direction': rec.get('direction'), 'target_jurisdiction': rec.get('target_jurisdiction'), 'other_jurisdiction': rec.get('other_jurisdiction'), 'question': rec.get('question') or rec.get('fact_pattern') or '', 'response_text': rec.get('response_text', ''), 'frame_label': label, 'silver_label': label, 'judge_confidence': conf, 'judge_rationale': rationale, 'judge_output_hash': hashlib.md5(text.encode()).hexdigest()[:12]}
            f.write(json.dumps(silver, ensure_ascii=False) + '\n')

def validate_judge_against_humans(silver_path: Path, human_csv: Path) -> dict:
    silver_by_id = {}
    for line in silver_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        silver_by_id[d['response_id']] = d
    pairs_eval = []
    with human_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row['response_id']
            silver = silver_by_id.get(rid)
            if not silver:
                continue
            silver_label = silver['silver_label']
            if silver_label not in LABELS:
                continue
            human_label_1 = (row.get('frame_label_annotator1') or '').strip().lower()
            human_label_2 = (row.get('frame_label_annotator2') or '').strip().lower()
            gold = human_label_1 if human_label_1 == human_label_2 else human_label_1 or human_label_2
            if gold not in LABELS:
                continue
            pairs_eval.append((silver_label, gold))
    if not pairs_eval:
        return {'n': 0, 'error': 'no matching ids between silver and human gold'}
    y_pred, y_true = zip(*pairs_eval)
    return {'n': len(pairs_eval), 'agreement': sum((a == b for a, b in pairs_eval)) / len(pairs_eval), 'cohen_kappa': cohen_kappa_score(y_true, y_pred, labels=LABELS), 'per_class': classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0), 'confusion_matrix': confusion_matrix(y_true, y_pred, labels=LABELS).tolist(), 'labels': LABELS}

def main():
    import sys
    if len(sys.argv) < 3:
        print('usage: python -m conceptlex.silver_label <responses_dir> <output_path> [validation_csv]')
        sys.exit(1)
    responses_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    validation_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    pairs = load_concept_pairs()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    merged_input = output_path.parent / 'all_responses_merged.jsonl'
    with merged_input.open('w') as out:
        for resp_file in sorted(responses_dir.glob('responses-*.jsonl')):
            for line in resp_file.read_text().splitlines():
                if not line.strip():
                    continue
                out.write(line + '\n')
    print(f'merged input -> {merged_input}')
    silver_label_responses(merged_input, pairs, output_path)
    print(f'silver labels written to {output_path}')
    if validation_csv and validation_csv.exists():
        report = validate_judge_against_humans(output_path, validation_csv)
        print('\n=== Judge vs. Human ===')
        print(json.dumps({k: v for k, v in report.items() if k != 'per_class'}, indent=2))
        if report.get('cohen_kappa', 0) < 0.6:
            print("\nWARNING: Cohen's kappa below 0.6 threshold; consider revising the judge prompt.")
if __name__ == '__main__':
    main()

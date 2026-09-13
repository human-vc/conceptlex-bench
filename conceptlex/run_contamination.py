import json
import multiprocessing as mp
import os
from pathlib import Path
from conceptlex.run_inference import MODELS, render_prompt
RETRIEVAL_K = 3

def load_retrieval_items(items_dir: Path) -> list[dict]:
    path = items_dir / 'retrieval_contamination.jsonl'
    if not path.exists():
        return []
    items = []
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            d['item_type'] = 'retrieval_contamination'
            items.append(d)
    return items

def expand_with_conditions(items: list[dict], retriever) -> list[dict]:
    expanded = []
    for item in items:
        question = item['question']
        target_jur = item['target_jurisdiction']
        conditions_meta = item.get('retrieval_conditions', [])
        other_jur = next((c['corpus_jurisdiction'] for c in conditions_meta if c['condition'] == 'mismatch'), None)
        empty_record = {**item, 'retrieval_condition': 'empty', 'corpus_jurisdiction': None, 'retrieved_context': None}
        expanded.append(empty_record)
        if target_jur and target_jur != '?':
            match_chunks = retriever.top_k(question, target_jur, k=RETRIEVAL_K)
            expanded.append({**item, 'retrieval_condition': 'match', 'corpus_jurisdiction': target_jur, 'retrieved_context': match_chunks})
        if other_jur and other_jur != '?':
            mismatch_chunks = retriever.top_k(question, other_jur, k=RETRIEVAL_K)
            expanded.append({**item, 'retrieval_condition': 'mismatch_topic', 'corpus_jurisdiction': other_jur, 'retrieved_context': mismatch_chunks})
            if hasattr(retriever, 'top_k_plausibility_matched'):
                pc_chunks = retriever.top_k_plausibility_matched(question, target_jur, other_jur, k=RETRIEVAL_K)
                if pc_chunks:
                    expanded.append({**item, 'retrieval_condition': 'mismatch_plausibility', 'corpus_jurisdiction': other_jur, 'retrieved_context': pc_chunks})
    return expanded

def _run_model_subprocess(model_spec: dict, items: list[dict], output_path: str):
    from vllm import LLM, SamplingParams
    kwargs = dict(model=model_spec['hf_id'], dtype='bfloat16', max_model_len=model_spec['max_model_len'], gpu_memory_utilization=model_spec.get('gpu_mem', 0.9), max_num_seqs=model_spec.get('max_num_seqs', 32), max_num_batched_tokens=8192, enable_prefix_caching=True, trust_remote_code=True, seed=42)
    if model_spec.get('quant'):
        kwargs['quantization'] = model_spec['quant']
    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()
    template_kwargs = {'tokenize': False, 'add_generation_prompt': True}
    if 'qwen_thinking' in model_spec:
        template_kwargs['enable_thinking'] = model_spec['qwen_thinking']
    prompts = []
    for item in items:
        ctx = item.get('retrieved_context')
        ctx_str = '\n\n'.join(ctx) if isinstance(ctx, list) else ctx or ''
        user_msg = render_prompt(item, ctx_str or None)
        chat = [{'role': 'user', 'content': user_msg}]
        try:
            rendered = tokenizer.apply_chat_template(chat, **template_kwargs)
        except TypeError:
            template_kwargs.pop('enable_thinking', None)
            rendered = tokenizer.apply_chat_template(chat, **template_kwargs)
        prompts.append(rendered)
    sampling = SamplingParams(temperature=0.0, max_tokens=800, top_p=1.0, seed=42)
    outputs = llm.generate(prompts, sampling)
    with open(output_path, 'w') as f:
        for item, out in zip(items, outputs):
            record = {'id': f"resp-{model_spec['name']}-{item['id']}-{item.get('retrieval_condition', '')}", 'item_id': item['id'], 'item_type': 'retrieval_contamination', 'model': model_spec['name'], 'retrieval_condition': item.get('retrieval_condition'), 'corpus_jurisdiction': item.get('corpus_jurisdiction'), 'language': item.get('language'), 'question': item.get('question'), 'target_jurisdiction': item.get('target_jurisdiction'), 'response_text': out.outputs[0].text, 'n_retrieved': len(item.get('retrieved_context') or []), 'decoding': {'temperature': 0.0, 'max_tokens': 800, 'top_p': 1.0, 'seed': 42}}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def run_model_isolated(model_spec: dict, items: list[dict], output_path: Path):
    ctx = mp.get_context('spawn')
    proc = ctx.Process(target=_run_model_subprocess, args=(model_spec, items, str(output_path)))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"Contamination inference failed for {model_spec['name']} (exit {proc.exitcode})")

def main():
    import sys
    if len(sys.argv) < 4:
        print('usage: python -m conceptlex.run_contamination <items_dir> <corpora_indexed_dir> <output_dir>')
        sys.exit(1)
    items_dir = Path(sys.argv[1])
    corpora_dir = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)
    from conceptlex.retriever import Retriever
    items = load_retrieval_items(items_dir)
    print(f'loaded {len(items)} retrieval_contamination items')
    retriever = Retriever(corpora_dir)
    expanded = expand_with_conditions(items, retriever)
    print(f'expanded to {len(expanded)} (item, condition) pairs')
    for spec in MODELS:
        out_path = output_dir / f"contamination-{spec['name']}.jsonl"
        if out_path.exists():
            print(f"skipping {spec['name']} (already exists)")
            continue
        print(f"running {spec['name']} contamination inference in isolated subprocess")
        run_model_isolated(spec, expanded, out_path)
if __name__ == '__main__':
    if not os.environ.get('HF_TOKEN') and (not os.environ.get('HUGGING_FACE_HUB_TOKEN')):
        print('note: meta-llama models require HF_TOKEN or HUGGING_FACE_HUB_TOKEN', flush=True)
    main()

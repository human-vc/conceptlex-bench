import json
import multiprocessing as mp
import os
from pathlib import Path
MODELS = [{'name': 'mistral-7b-instruct-v0.3', 'hf_id': 'mistralai/Mistral-7B-Instruct-v0.3', 'quant': None, 'max_model_len': 8192, 'max_num_seqs': 64, 'gpu_mem': 0.9}, {'name': 'saullm-7b-instruct', 'hf_id': 'Equall/Saul-7B-Instruct-v1', 'quant': None, 'max_model_len': 8192, 'max_num_seqs': 64, 'gpu_mem': 0.9}, {'name': 'llama-3-8b-instruct', 'hf_id': 'meta-llama/Meta-Llama-3-8B-Instruct', 'quant': None, 'max_model_len': 8192, 'max_num_seqs': 64, 'gpu_mem': 0.9}, {'name': 'lawma-8b', 'hf_id': 'ricdomolm/lawma-8b', 'quant': None, 'max_model_len': 8192, 'max_num_seqs': 64, 'gpu_mem': 0.9}, {'name': 'qwen3-32b', 'hf_id': 'Qwen/Qwen3-32B', 'quant': None, 'max_model_len': 8192, 'max_num_seqs': 16, 'gpu_mem': 0.92, 'qwen_thinking': False}, {'name': 'llama-3.3-70b-awq', 'hf_id': 'ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4', 'quant': None, 'max_model_len': 8192, 'max_num_seqs': 16, 'gpu_mem': 0.9}]

def load_items(items_dir: Path) -> list[dict]:
    items = []
    for fname in ('mcq.jsonl', 'generative.jsonl', 'retrieval_contamination.jsonl'):
        path = items_dir / fname
        if not path.exists():
            continue
        item_type = fname.split('.')[0]
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            item['item_type'] = item_type
            items.append(item)
    return items

def render_prompt(item: dict, retrieval_context: str | None=None) -> str:
    if item['item_type'] == 'mcq':
        opts = '\n'.join((f'({chr(65 + i)}) {o}' for i, o in enumerate(item['options'])))
        ctx = f'\n\nContext:\n{retrieval_context}\n' if retrieval_context else ''
        return f"{item['question']}{ctx}\n\n{opts}\n\nProvide your answer letter (A, B, C, or D) and a brief justification."
    if item['item_type'] == 'generative':
        return item['fact_pattern']
    ctx = f'\n\nContext:\n{retrieval_context}\n' if retrieval_context else ''
    return f"{item['question']}{ctx}\n\nProvide a substantive legal analysis."

def expand_retrieval_items(retrieval_items: list[dict], retriever) -> list[dict]:
    expanded = []
    for item in retrieval_items:
        for cond in item['retrieval_conditions']:
            sub = {**item, 'retrieval_condition': cond['condition'], 'corpus_jurisdiction': cond['corpus_jurisdiction']}
            if cond['corpus_jurisdiction'] is None:
                sub['retrieved_context'] = None
            else:
                sub['retrieved_context'] = retriever.top_k(item['question'], cond['corpus_jurisdiction'], k=3)
            expanded.append(sub)
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
            record = {'id': f"resp-{model_spec['name']}-{item['id']}-{item.get('retrieval_condition', '')}", 'item_id': item['id'], 'item_type': item['item_type'], 'model': model_spec['name'], 'retrieval_condition': item.get('retrieval_condition'), 'corpus_jurisdiction': item.get('corpus_jurisdiction'), 'language': item.get('language'), 'question': item.get('question') or item.get('fact_pattern'), 'options': item.get('options'), 'correct_idx': item.get('correct_idx'), 'response_text': out.outputs[0].text, 'prompt_template_version': 'v2', 'decoding': {'temperature': 0.0, 'max_tokens': 800, 'top_p': 1.0, 'seed': 42}}
            f.write(json.dumps(record) + '\n')

def run_model_isolated(model_spec: dict, items: list[dict], output_path: Path):
    ctx = mp.get_context('spawn')
    proc = ctx.Process(target=_run_model_subprocess, args=(model_spec, items, str(output_path)))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"Inference failed for {model_spec['name']} with exit code {proc.exitcode}")

def main():
    import sys
    if len(sys.argv) < 3:
        print('usage: python -m conceptlex.run_inference <items_dir> <output_dir> [retriever_path]')
        sys.exit(1)
    items_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    retriever_path = sys.argv[3] if len(sys.argv) > 3 else None
    items = load_items(items_dir)
    if retriever_path:
        from conceptlex.retriever import Retriever
        retriever = Retriever(retriever_path)
        retrieval_items = [i for i in items if i['item_type'] == 'retrieval_contamination']
        other_items = [i for i in items if i['item_type'] != 'retrieval_contamination']
        items = other_items + expand_retrieval_items(retrieval_items, retriever)
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in MODELS:
        out_path = output_dir / f"responses-{spec['name']}.jsonl"
        if out_path.exists():
            print(f"skipping {spec['name']} (already exists)")
            continue
        print(f"running {spec['name']} in isolated subprocess")
        run_model_isolated(spec, items, out_path)
if __name__ == '__main__':
    if not os.environ.get('HF_TOKEN') and (not os.environ.get('HUGGING_FACE_HUB_TOKEN')):
        print('note: meta-llama models require HF_TOKEN or HUGGING_FACE_HUB_TOKEN', flush=True)
    main()

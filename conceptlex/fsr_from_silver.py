from __future__ import annotations
import json
import collections
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
SHIFT_STRICT = {'pure_other'}
SHIFT_BROAD = {'pure_other', 'hybrid'}
GENERATIVE_EXCLUDE = {'lawma-8b'}
LLAMA_FAMILY = {'llama-3.3-70b-awq', 'llama-3-8b-instruct', 'lawma-8b'}

def condition_of(response_id: str) -> str:
    if response_id.endswith('mismatch_topic'):
        return 'mismatch_topic'
    if response_id.endswith('mismatch_plausibility'):
        return 'mismatch_plausibility'
    if response_id.endswith('-match'):
        return 'match'
    if response_id.endswith('empty'):
        return 'empty'
    return '?'

def assert_frame_labels_ok(labels: list, source: Path) -> None:
    distinct = {l for l in labels if l}
    if len(distinct) < 2:
        raise ValueError(f"{source}: frame_label collapsed to {distinct or 'nothing'} across {len(labels)} contamination records. This is the signature of the degenerate mDeBERTa classifier ablation (classifier_ablation-judged-*.jsonl), not the LLM-judge scoring labels. Pass silver_labels_v3.jsonl.")

def load_silver(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get('item_type') != 'retrieval_contamination':
            continue
        d['_cond'] = condition_of(d.get('response_id', ''))
        out.append(d)
    assert_frame_labels_ok([d.get('frame_label') for d in out], path)
    return out

def load_equivalence(pairs_path: Path) -> dict[tuple[str, str], str]:
    eq = {}
    for line in pairs_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        for direction in ('a_to_b', 'b_to_a'):
            label = d.get('labels', {}).get(direction, {}).get('equivalence')
            if label:
                eq[d['id'], direction] = label
    return eq

def bca_ci(values, iters=10000, alpha=0.05, seed=42):
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2 or np.allclose(arr, arr[0]):
        m = float(arr.mean()) if len(arr) else 0.0
        return (m, m)
    rng = np.random.default_rng(seed)
    try:
        r = bootstrap((arr,), np.mean, n_resamples=iters, confidence_level=1 - alpha, method='BCa', rng=rng)
    except TypeError:
        r = bootstrap((arr,), np.mean, n_resamples=iters, confidence_level=1 - alpha, method='BCa', random_state=rng)
    return (float(r.confidence_interval.low), float(r.confidence_interval.high))

def per_pair_fsr(records: list[dict]) -> list[dict]:
    grp = collections.defaultdict(dict)
    for r in records:
        key = (r.get('model'), r.get('concept_pair_id'), r.get('direction'))
        grp[key][r['_cond']] = r.get('frame_label')
    rows = []
    for (model, pair, direction), conds in grp.items():
        empty = conds.get('empty')
        mismatch = conds.get('mismatch_topic')
        if empty is None or mismatch is None:
            continue
        row = {'model': model, 'concept_pair_id': pair, 'direction': direction, 'fsr_strict': int(mismatch in SHIFT_STRICT) - int(empty in SHIFT_STRICT), 'fsr_broad': int(mismatch in SHIFT_BROAD) - int(empty in SHIFT_BROAD), 'target_loss': int(empty == 'pure_target') - int(mismatch == 'pure_target')}
        plaus = conds.get('mismatch_plausibility')
        if plaus is not None:
            row['fsr_strict_plaus'] = int(plaus in SHIFT_STRICT) - int(empty in SHIFT_STRICT)
            row['fsr_broad_plaus'] = int(plaus in SHIFT_BROAD) - int(empty in SHIFT_BROAD)
        rows.append(row)
    return rows

def aggregate(rows: list[dict], group_field: str | None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    groups = [None] if group_field is None else sorted(df[group_field].dropna().unique())
    for g in groups:
        sub = df if g is None else df[df[group_field] == g]
        rec = {'group': 'ALL' if g is None else g, 'n_pairs': len(sub)}
        for metric in ('fsr_strict', 'fsr_broad', 'target_loss'):
            vals = sub[metric].tolist()
            lo, hi = bca_ci(vals)
            rec[f'{metric}_mean'] = float(np.mean(vals)) if vals else 0.0
            rec[f'{metric}_lo'] = lo
            rec[f'{metric}_hi'] = hi
        out.append(rec)
    return pd.DataFrame(out)

def main():
    import sys
    silver_path = Path(sys.argv[1])
    pairs_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('data/concept_pairs.jsonl')
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('outputs/results_v2')
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_silver(silver_path)
    eq = load_equivalence(pairs_path)
    rows = per_pair_fsr(records)
    for r in rows:
        r['equivalence'] = eq.get((r['concept_pair_id'], r['direction']), 'unknown')
    gen_rows = [r for r in rows if r['model'] not in GENERATIVE_EXCLUDE]
    nonllama_rows = [r for r in gen_rows if r['model'] not in LLAMA_FAMILY]
    by_model = aggregate(rows, 'model')
    by_eq_all = aggregate(rows, 'equivalence')
    by_eq_gen = aggregate(gen_rows, 'equivalence')
    overall_all = aggregate(rows, None)
    overall_gen = aggregate(gen_rows, None)
    overall_nonllama = aggregate(nonllama_rows, None)
    by_model.to_csv(out_dir / 'fsr_by_model.csv', index=False)
    by_eq_gen.to_csv(out_dir / 'fsr_by_equivalence.csv', index=False)
    overall_gen.to_csv(out_dir / 'fsr_overall.csv', index=False)
    pd.DataFrame(rows).to_csv(out_dir / 'fsr_per_pair.csv', index=False)
    ct = pd.DataFrame(gen_rows).pivot_table(index='model', columns='equivalence', values='fsr_broad', aggfunc='mean')
    ct.to_csv(out_dir / 'fsr_model_by_equivalence.csv')
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 24)
    print('=== FSR by model (all 6, silver-label judge) ===')
    print(by_model.to_string(index=False))
    print('\n=== FSR overall (generative models only, Lawma excluded) ===')
    print(overall_gen.to_string(index=False))
    print('\n=== FSR overall (cross-family robustness: non-Llama subjects only, judge=Llama-3.3-70B) ===')
    print(overall_nonllama.to_string(index=False))
    overall_nonllama.to_csv(out_dir / 'fsr_overall_nonllama.csv', index=False)
    print('\n=== FSR by Sarcevic class — ALL models ===')
    print(by_eq_all.to_string(index=False))
    print('\n=== FSR by Sarcevic class — generative only (Lawma excluded) ===')
    print(by_eq_gen.to_string(index=False))
    print('\n=== broad-FSR crosstab: model x equivalence (generative only) ===')
    print(ct.round(3).to_string())
    gdf = pd.DataFrame(gen_rows)
    if 'fsr_broad_plaus' in gdf.columns and gdf['fsr_broad_plaus'].notna().any():
        pl = gdf.dropna(subset=['fsr_broad_plaus'])
        s_lo, s_hi = bca_ci(pl['fsr_strict_plaus'].tolist())
        b_lo, b_hi = bca_ci(pl['fsr_broad_plaus'].tolist())
        print('\n=== Plausibility-matched mismatch (controls relevance magnitude; generative only) ===')
        print(f"  fsr_strict_plaus mean={pl['fsr_strict_plaus'].mean():.3f} [{s_lo:.3f}, {s_hi:.3f}]")
        print(f"  fsr_broad_plaus  mean={pl['fsr_broad_plaus'].mean():.3f} [{b_lo:.3f}, {b_hi:.3f}]  (n={len(pl)})")
        print('  -> if comparable to topic-mismatch, the effect is jurisdictional capture, not topical distraction')
    else:
        print('\n(no mismatch_plausibility condition in this run; rerun contamination after adding the retriever method)')
if __name__ == '__main__':
    main()

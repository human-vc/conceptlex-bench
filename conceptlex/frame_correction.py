import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from conceptlex.judge_prompts import LABELS
ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / 'data' / 'human_validation'
RUNS = {'mitigation_labels': 'DE-US mitigation (31 pairs, 2 models)', 'direction_split_labels': 'direction split (31 pairs, 2 models)', 'equivalence_labels': 'equivalence (63 pairs, 2 models)'}
STRICT = {'pure_other'}
BROAD = {'pure_other', 'hybrid'}

def _column(path, field):
    with open(path, encoding='utf-8', newline='') as handle:
        return {r['item_id']: r[field].strip() for r in csv.DictReader(handle) if r[field].strip()}

def load_pool(labels_dir):
    pool = []
    for run in RUNS:
        path = Path(labels_dir) / f'{run}.jsonl'
        if not path.exists():
            raise SystemExit(f'missing {path}; run the replication scripts first')
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                rec = json.loads(line)
                if rec['frame_label'] in LABELS:
                    pool.append({'run': run, 'yhat': rec['frame_label'], 'cond': rec['retrieval_condition'], 'pair': rec['concept_pair_id']})
    return pool

def load_gold():
    a = _column(GOLD_DIR / 'annotator_A.csv', 'label')
    b = _column(GOLD_DIR / 'annotator_B.csv', 'label')
    consensus = {i: a[i].lower() for i in set(a) & set(b) if a[i] == b[i]}
    with open(GOLD_DIR / 'scoring_key.csv', encoding='utf-8', newline='') as handle:
        key = {r['item_id']: r for r in csv.DictReader(handle)}
    return [{'yhat': key[i]['judge_label'], 'y': consensus[i], 'pair': key[i]['concept_pair_id'], 'cond': key[i]['retrieval_condition']} for i in consensus if i in key]

def group(condition):
    return 'mismatch' if condition.startswith('mismatch') else 'other'

def confusion(rows, split=False, min_cell=5):

    def build(subset):
        counts = defaultdict(Counter)
        for r in subset:
            counts[r['yhat']][r['y']] += 1
        out = {}
        for k in LABELS:
            total = sum(counts[k].values())
            out[k] = {c: counts[k][c] / total for c in LABELS} if total >= min_cell else None
        return out
    matrices = {'pooled': build(rows)}
    if split:
        for g in ('mismatch', 'other'):
            matrices[g] = build([r for r in rows if group(r['cond']) == g])
    return matrices

def corrected(pool, matrices, target, run, cond):
    subset = [p for p in pool if p['run'] == run and p['cond'] == cond]
    if not subset:
        return None
    key = group(cond) if group(cond) in matrices else 'pooled'
    total, n = (0.0, len(subset))
    for label, count in Counter((p['yhat'] for p in subset)).items():
        cell = matrices.get(key, {}).get(label) or matrices['pooled'].get(label)
        if cell is None:
            return None
        total += count / n * sum((cell[c] for c in target))
    return total

def naive(pool, target, run, cond):
    subset = [p for p in pool if p['run'] == run and p['cond'] == cond]
    return sum((1 for p in subset if p['yhat'] in target)) / len(subset) if subset else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', default=str(ROOT / 'data' / 'replication'))
    parser.add_argument('--sensitivity', action='store_true')
    parser.add_argument('--boot', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=20260810)
    args = parser.parse_args()
    random.seed(args.seed)
    pool, gold = (load_pool(args.labels), load_gold())
    print(f'pool N = {len(pool)}, gold n = {len(gold)}')
    print(f"gold per stratum: {dict(Counter((g['yhat'] for g in gold)))}\n")
    base = confusion(gold, split=args.sensitivity)
    print('P(human | judge)')
    header = ''.join((f'{c[:9]:>11s}' for c in LABELS))
    print(f"  {'judge':16s}{header}      n")
    for k in LABELS:
        cell = base['pooled'][k]
        n = sum((1 for g in gold if g['yhat'] == k))
        if cell:
            print(f'  {k:16s}' + ''.join((f'{cell[c]:11.3f}' for c in LABELS)) + f' {n:6d}')
    pairs_by_run = defaultdict(list)
    for p in pool:
        pairs_by_run[p['run'], p['pair']].append(p)
    cells = defaultdict(list)
    for g in gold:
        cells[group(g['cond']) if args.sensitivity else 'all', g['yhat']].append(g)

    def resample():
        keys = defaultdict(list)
        for k in pairs_by_run:
            keys[k[0]].append(k)
        boot_pool = []
        for _, ks in keys.items():
            boot_pool += [x for k in random.choices(ks, k=len(ks)) for x in pairs_by_run[k]]
        boot_gold = []
        for rows in cells.values():
            by_pair = defaultdict(list)
            for r in rows:
                by_pair[r['pair']].append(r)
            ks = list(by_pair)
            boot_gold += [x for k in random.choices(ks, k=len(ks)) for x in by_pair[k]]
        return (boot_pool, confusion(boot_gold, split=args.sensitivity))
    label = 'condition-dependent' if args.sensitivity else 'pooled'
    print(f'\nframe-shift rate vs the empty baseline of the same run ({label} confusion)')
    for run in RUNS:
        conds = sorted({p['cond'] for p in pool if p['run'] == run and p['cond'] != 'empty'})
        if not conds:
            continue
        print(f'\n  {RUNS[run]}')
        print(f"    {'condition':32s}{'strict':>8s}{'95% CI':>19s}{'broad':>8s}{'95% CI':>19s}")
        for cond in conds:
            line = f'    {cond:32s}'
            for target in (STRICT, BROAD):
                point = corrected(pool, base, target, run, cond) - corrected(pool, base, target, run, 'empty')
                draws = []
                for _ in range(args.boot):
                    bp, bm = resample()
                    hi = corrected(bp, bm, target, run, cond)
                    lo = corrected(bp, bm, target, run, 'empty')
                    if hi is not None and lo is not None:
                        draws.append(hi - lo)
                draws.sort()
                a, b = (draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))])
                mark = '' if a <= 0 <= b else '*'
                line += f'{point:8.3f}  [{a:+.3f},{b:+.3f}]{mark:1s}'
            print(line)
if __name__ == '__main__':
    main()

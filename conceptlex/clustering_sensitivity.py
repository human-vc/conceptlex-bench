from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2
import statsmodels
import statsmodels.formula.api as smf
ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_BOOT = 10000

def read_jsonl(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines() if line.strip()]

def main():
    labels = read_jsonl('data/replication/equivalence_labels.jsonl')
    pairs = {p['id']: p for p in read_jsonl('data/concept_pairs.jsonl')}
    rows = []
    for r in labels:
        assert r['retrieval_condition'] in ('empty', 'mismatch_topic')
        assert r['frame_label'] in {'pure_target', 'pure_other', 'hybrid', 'shared_general', 'meta_refusal'}
        pair, direction = (r['concept_pair_id'], r['direction'])
        rows.append(dict(pair=pair, directed=pair + '|' + direction, direction=direction, model=r['model'], condition=r['retrieval_condition'], equivalence=pairs[pair]['labels'][direction]['equivalence'], strict=int(r['frame_label'] == 'pure_other'), broad=int(r['frame_label'] in ('pure_other', 'hybrid'))))
    df = pd.DataFrame(rows)
    df['condition'] = pd.Categorical(df['condition'], categories=['empty', 'mismatch_topic'])
    assert len(df) == 504 and df['pair'].nunique() == 63 and (df['directed'].nunique() == 126)
    assert not df.duplicated(['pair', 'direction', 'model', 'condition']).any()
    result = dict(scope='Released equivalence arm only; two models; not the five-model main run', seed=SEED, bootstrap_draws=N_BOOT, versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, statsmodels=statsmodels.__version__), input_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('data/replication/equivalence_labels.jsonl', 'data/concept_pairs.jsonl')}, n_responses=len(df), n_pairs=63, n_directed_items=126, equivalence_response_counts={str(k): int(v) for k, v in df.equivalence.value_counts().items()}, outcomes={})
    for outcome in ('strict', 'broad'):
        table = df.pivot(index=['pair', 'direction', 'model'], columns='condition', values=outcome)
        assert table.notna().all().all()
        delta = table['mismatch_topic'] - table['empty']
        p0, p1 = (float(table['empty'].mean()), float(table['mismatch_topic'].mean()))
        out = dict(empty_raw=p0, mismatch_raw=p1, fsr=float(delta.mean()), bootstrap={}, regression={})
        reference_var = (p0 * (1 - p0) + p1 * (1 - p1)) / len(table)
        for unit, levels in [('directed', ['pair', 'direction']), ('bilateral', ['pair'])]:
            sizes = delta.groupby(level=levels).size()
            assert sizes.nunique() == 1
            cluster_means = delta.groupby(level=levels).mean().to_numpy(dtype=float)
            rng = np.random.default_rng(SEED)
            draws = cluster_means[rng.integers(0, len(cluster_means), size=(N_BOOT, len(cluster_means)))].mean(axis=1)
            exact_boot_variance = float(np.var(cluster_means, ddof=0) / len(cluster_means))
            out['bootstrap'][unit] = dict(clusters=len(cluster_means), percentile_ci95=np.quantile(draws, [0.025, 0.975]).tolist(), se=float(np.std(draws, ddof=1)), design_effect_vs_independent_binomial=exact_boot_variance / reference_var)
        for unit, group_column in [('directed', 'directed'), ('bilateral', 'pair')]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    fitted = smf.logit(f'{outcome} ~ C(model) + C(condition) + C(equivalence)', data=df).fit(disp=False, method='lbfgs', maxiter=300, cov_type='cluster', cov_kwds={'groups': df[group_column]})
                    term = 'C(condition)[T.mismatch_topic]'
                    out['regression'][unit] = dict(converged=bool(fitted.mle_retvals.get('converged', False)), odds_ratio=float(np.exp(fitted.params[term])), or_ci95=np.exp(fitted.conf_int().loc[term]).tolist(), condition_log_odds_se=float(fitted.bse[term]), note='Additive model specification used for the released code main-effect OR.')
                except Exception as exc:
                    out['regression'][unit] = dict(error=repr(exc))
                out['regression'][unit]['warnings'] = [str(w.message) for w in caught]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                full = smf.logit(f'{outcome} ~ C(model) + C(condition) * C(equivalence)', data=df).fit(disp=False, method='lbfgs', maxiter=300)
                reduced = smf.logit(f'{outcome} ~ C(model) + C(condition) + C(equivalence)', data=df).fit(disp=False, method='lbfgs', maxiter=300)
                stat, dof = (float(2 * (full.llf - reduced.llf)), int(full.df_model - reduced.df_model))
                out['ordinary_lrt'] = dict(statistic=stat, df=dof, p=float(chi2.sf(stat, dof)), full_converged=bool(full.mle_retvals.get('converged', False)), reduced_converged=bool(reduced.mle_retvals.get('converged', False)), note='Ordinary LRT only. Not cluster-adjusted and not a substitute for the main-run interaction.')
            except Exception as exc:
                out['ordinary_lrt'] = dict(error=repr(exc))
            out['ordinary_lrt']['warnings'] = [str(w.message) for w in caught]
        result['outcomes'][outcome] = out
    (ROOT / 'outputs').mkdir(exist_ok=True)
    (ROOT / 'outputs' / 'clustering_sensitivity.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
if __name__ == '__main__':
    main()

# Replication artifacts

Labels and results for the three follow-up analyses reported in the paper's appendices.
Each was generated after the main run, on a local llama.cpp backend with reduced retrieval
corpora, so the absolute rates sit above those in the main tables. In every case the
within-run contrast is the quantity reported, not the level.

The judge, rubric, few-shot exemplars, and decoding parameters are the same as the main
labeling run in all three.

| File | What it is |
|---|---|
| `judge_validation_labels.jsonl` | Judge labels for the 150 human-annotated validation responses, used for the judge-versus-human agreement in the judge-validation appendix. Pair with `conceptlex_validation_set.annotated.csv`, which holds the two human annotators' labels. |
| `direction_split_labels.jsonl` | 248 labelled responses, 31 German--United States pairs, both directions, empty and topic-mismatch conditions, SaulLM-7B-Instruct and Mistral-7B-Instruct-v0.3. |
| `direction_split_result.txt` | Output of `conceptlex.direction_split` on the above. |
| `mitigation_labels.jsonl` | 744 labelled responses, same pairs, six conditions: no-retrieval baseline, correct-jurisdiction match, both mismatch arms, and both mismatch arms with the jurisdiction-anchoring instruction. |
| `mitigation_result.txt` | Output of `conceptlex.mitigation_ci` on the above, with concept-pair cluster bootstrap intervals. |
| `equivalence_labels.jsonl` | 504 labelled responses over all 63 pairs and four target jurisdictions, no-retrieval and topic-mismatch conditions, same two models. |

## Reproducing

```
python -m conceptlex.direction_split data/replication/direction_split_labels.jsonl data/concept_pairs.jsonl
python -m conceptlex.mitigation_ci   data/replication/mitigation_labels.jsonl
python -m conceptlex.frame_correction
python -m conceptlex.frame_correction --sensitivity
```

Regenerating the responses themselves needs the corpora and a served model:

```
python -m conceptlex.build_corpora_local outputs/corpora_local --docs 1200 --max-chunks 6000
bash scripts/run_direction_split.sh
bash scripts/run_mitigation_ci.sh
```

## Not included

An equivalence-class replication was also run and is not reported in the paper. It covered
two models rather than five on the same reduced corpora, and it did not reproduce the
ordering the main run's point estimates suggest (near-equivalence 0.438, non-equivalence
0.500; near minus non was -0.062 with a 95 percent interval of -0.240 to 0.101). Because
its scope differs from the main run, it cannot adjudicate that question either way, and
the paper's position remains that the class analysis is exploratory and underpowered. The
code (`conceptlex.equivalence_generate`, `conceptlex.equivalence_test`) is released so the
analysis can be run at full scale.

# ConceptLex

A benchmark measuring whether a legal language model keeps reasoning in the jurisdiction
it was asked about when retrieval supplies authority from another legal system.

## Install

```bash
pip install -r requirements.txt
```

Steps 2-5 need a GPU (vLLM). Gated models need `export HF_TOKEN=...`.
Run everything from the repository root.

## Run

```bash
# 1. build items from the concept-pair seed list
python -m conceptlex.generate_items data/concept_pairs.jsonl outputs/items

# 2. build per-jurisdiction retrieval corpora
python -m conceptlex.build_corpora       outputs/corpora 30000
python -m conceptlex.build_corpora_extra outputs/corpora 15000

# 3. multiple-choice and generative inference
python -m conceptlex.run_inference outputs/items outputs/responses

# 4. retrieval conditions: empty, match, mismatch_topic, mismatch_plausibility
python -m conceptlex.run_contamination outputs/items outputs/corpora outputs/responses

# 5. judge labels (set CONCEPTLEX_JUDGE_MODEL for the cross-family check)
python -m conceptlex.silver_label outputs/responses outputs/labels.jsonl

# 6. frame-shift rates
python -m conceptlex.fsr_from_silver outputs/labels.jsonl data/concept_pairs.jsonl outputs/results
```

Step 4 writes `contamination-*.jsonl`; step 5 reads `responses-*.jsonl`. Rename or
symlink between them.

## Run without a GPU

These use only the files in `data/` and need no inference:

```bash
python -m conceptlex.frame_correction        # judge-error correction (--sensitivity, --boot N)
python -m conceptlex.clustering_sensitivity  # clustering sensitivity on the equivalence arm
```

## Data

- `data/concept_pairs.jsonl` — 63 bilateral concept pairs with equivalence labels and distractors
- `data/replication/` — response-level judge labels for the equivalence, direction and mitigation runs
- `data/human_validation/` — annotator files, scoring key, codebook

The main run's per-model labels are not part of the release.

## License

Code MIT (`LICENSE`). Retrieval corpora are rebuilt from public sources by the builders
rather than redistributed, and retain their own licenses.

# Human validation set

The prediction-stratified human annotation used to measure judge error and to
correct the reported frame-shift rates. This is the second validation set. The
first, described in the judge-validation appendix, was drawn from a population in
which the two classes that define the frame-shift rate barely occur, so it could
not speak to them.

## Sampling

140 responses were drawn from the 1,494 judged responses of the three replication
runs, stratified on the judge's own predicted label and balanced across target
jurisdictions within each stratum. Responses shorter than 80 characters were
excluded as unclassifiable.

| judge label | drawn | available in pool |
|---|---|---|
| `pure_other` | 40 | 236 |
| `hybrid` | 40 | 168 |
| `pure_target` | 30 | 1,034 |
| `meta_refusal` | 15 | 40 |
| `shared_general` | 15 | 16 |

Because the sample is stratified on the judge's prediction, these proportions are
not population base rates. Raw agreement computed on this file is a per-class
measure, not an estimate of accuracy on the benchmark. What the design does
identify is the conditional distribution of the human label given the judge label,
which is what the correction uses.

## Files

| File | What it is |
|---|---|
| `annotator_A.csv` | First annotator's labels, confidence, and notes, one row per item, including the response text as shown to the annotator |
| `annotator_B.csv` | Second annotator's labels, same structure |
| `scoring_key.csv` | Judge label, retrieval condition, model, source run, response id and concept pair id per item. Withheld from both annotators until their files were returned |
| `CODING_INSTRUCTIONS.md` | The codebook: five labels, signals, and the six rules that decide the hard cases |
| `BACKGROUND.md` | Study context given to annotators before coding |
| `WORKED_EXAMPLES.md` | Constructed illustrations of each label. These are hand-written, not items from the sample, so that no annotator judgement was anchored to a machine label |

Annotators worked independently and blind to the retrieval condition, the
generating model, and the judge's label. They agreed at 0.950 raw, Cohen's
kappa 0.924, Gwet's AC1 0.940, reaching consensus on 133 of 140 items. Consensus
is defined as an exact match between the two labels; the seven disagreements are
left unresolved rather than adjudicated, and are excluded from the correction.

## Reproducing

The correction reads this directory and the replication run outputs:

```
python -m conceptlex.frame_correction
python -m conceptlex.frame_correction --sensitivity
```

The first prints the judge confusion matrix and the corrected frame-shift rates
with concept-pair cluster bootstrap intervals, contrasting each condition with the
empty baseline of its own run. The second relaxes the assumption that judge error
is independent of retrieval condition, re-estimating the matrix separately for
mismatch and non-mismatch conditions and falling back to the pooled cell wherever
a cell holds fewer than five gold items. The strict rate is stable under that
relaxation and the broad rate is not, for the structural reason that the
`pure_other` and `hybrid` strata contain almost no non-mismatch gold items.

To test the judge variants reported alongside the correction, including the
reduced taxonomy:

```
CONCEPTLEX_API_KEY=... python -m conceptlex.llama_4label_test \
  data/human_validation outputs/llama4
```

## Judge variant labels

`llama_five.jsonl` and `llama_four.jsonl` are Llama-3.3-70B relabelings of the gold
items under the deployed five-label prompt and with `shared_general` removed, used
for the judge-variant table. They were produced by the command above. Note that
these disagree with the stored labels in `scoring_key.csv` on 13 percent of items
despite identical prompt and decoding, the difference being a local quantization
in the main run against a hosted endpoint here.

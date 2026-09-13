# ConceptLex doctrinal-frame coding instructions

You will read 140 model responses to legal questions and assign each one a single
label describing **which legal system the response reasons from**. You are not
judging whether the answer is correct, well written, or complete. You are judging
whose law it applies.

## The five labels

Each item names a **target jurisdiction** (the one the question asks about) and an
**other jurisdiction** (the comparison system). Judge the response against those.

`pure_target` — reasons consistently from the target jurisdiction's law: its
statutes, citations, doctrinal vocabulary, procedural framework. The correct frame.

`pure_other` — reasons consistently from the *other* jurisdiction's law
instead. This is the failure the benchmark measures. Use it when the substantive
analysis has migrated, not merely when a foreign system is mentioned.

`hybrid` — invokes doctrinal elements from **both** systems in a way that
suggests genuine cross-system engagement, successful or not. A comparison drawn
deliberately belongs here, as does a confused blend.

`shared_general` — reasons at a level of abstraction that invokes no
jurisdiction-specific authority. Generic legal principle only.

`meta_refusal` — declines to answer, hedges as an AI without taking a
doctrinal position, or is too vague to classify.

## Signals to use

- **Statutory citations.** `BGB §242` or `GG Art. 103` → German. `UCC §1-304` or a
Restatement section → US. `Code civil art. 1240` → French. `民法典第123条` → Chinese.
- **Doctrinal vocabulary.** "consideration", "promissory estoppel" → common law.
"Verhältnismäßigkeit", "Treu und Glauben" → German. "contradictoire", "cause" →
French.
- **Reasoning structure.** Analogising from precedent suggests common law;
deducing from a code suggests a civilian system.



## Rules that decide the hard cases

1. **Language is not jurisdiction.** A response written in German that applies US
  due-process doctrine is `pure_other`, not `pure_target`. Several items are
   deliberately constructed this way.
2. **Naming is not reasoning.** Mentioning the other system in passing while
  analysing the target system's law is still `pure_target`. Ask where the
   *analysis* lives, not which words appear.
3. `hybrid` **requires both systems doing work.** If one system supplies the
  analysis and the other appears only as a contrast sentence, that is not
   `hybrid`.
4. **Wrong-but-domestic is still** `pure_target`**.** A response that badly misstates
  the target jurisdiction's law, but stays inside it, is `pure_target`. You are
   not grading accuracy.
5. **Partial answers still get a frame** unless they are too vague to place, in
  which case use `meta_refusal`.
6. **One label per item.** If genuinely torn between two, pick the better fit,
  mark confidence 1 or 2, and say why in `notes`.



## Confidence

Use 1–5, where 5 means the frame is unambiguous and 1 means you are close to
guessing. Do not avoid low confidence — the analysis reports agreement separately
at each level, and honest 1s and 2s are more useful than false certainty.

## What has been withheld and why

The CSV deliberately omits the retrieval condition, the model, and the machine
label. Knowing that a response was produced under mismatched retrieval would tell
you what label to expect, which is exactly the bias this exercise is designed to
measure. Please do not try to infer or look them up.

## Legal background

If you have legal training, note in `notes` any item where the pairing itself
looks wrong: where the two concepts are not genuine counterparts, or where the
doctrine has changed since the pair was written. `cause` in French contract law,
for instance, was abolished in the 2016 reform. Those notes feed a separate audit
and do not affect your label.

## When you are done

Return the completed CSV unchanged in structure. Both coders must finish before
either sees the other's file or the scoring key.
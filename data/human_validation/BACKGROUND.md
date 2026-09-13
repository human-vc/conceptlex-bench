# Background: what this study is and why your labels matter

Read this before `CODING_INSTRUCTIONS.md`. It takes about five minutes and assumes
no familiarity with the project.

## The question

Legal AI assistants increasingly answer questions by retrieving documents and
reasoning over them. That works when the retrieved document belongs to the legal
system the user is asking about. It is less obviously safe when it does not.

A fact is the same fact wherever you retrieve it. A legal doctrine is not. The
right to be heard before an adverse decision is purely procedural in German
constitutional law, where it is called *rechtliches Gehör*. In the United States
the nearest counterpart, procedural due process, is broader and sits alongside a
substantive branch with no German analogue. Hand a model a question about German
law together with an American case, and it may quietly answer as though the
American doctrine governed.

This study measures how often that happens.

## How the data you are reading was produced

The benchmark is built from **concept pairs**: two doctrines from different legal
systems that are counterparts, like *consideration* and *cause*, or *rechtliches
Gehör* and *due process*. Each pair is read in both directions, so the same pair
yields one question about German law and one about American law.

For each question the model was given one of several kinds of context: none at
all, documents from the jurisdiction actually asked about, or documents from the
*other* jurisdiction. You are reading the answers it produced.

This is why every row has a **target jurisdiction**, the one the question asks
about, and an **other jurisdiction**, its counterpart in the pair. Your job is to
say which of the two the answer actually reasons from.

## Why humans are needed

Every result in the paper depends on these labels, and until now they were
assigned by another language model acting as a judge. That judge was checked
against human annotation once, but the sample it was checked on happened to
contain almost no examples of the two labels that matter most: cases where the
model reasoned from the wrong jurisdiction, and cases where it blended the two.
Agreement was high, but almost entirely on easy categories.

So the headline finding rests on a classifier whose accuracy on the decisive
categories has never actually been measured. That is the gap you are closing.

The 140 items you will see were deliberately selected so that those two categories
are well represented. They are **not** a random sample of the benchmark, and the
proportions you see do not reflect how often each outcome occurs in the study.
Please do not calibrate to what seems frequent in the file.

## What happens to your labels

Two coders label the same items independently. We first measure whether the two of
you agree with each other, because comparing a machine to a human is only
meaningful if humans agree among themselves. On the items where you both give the
same label, we then measure how often the machine judge matches, reported
separately for each category.

If the machine turns out to be accurate on the decisive categories, the paper's
results stand and can say so with evidence. If it does not, that changes what the
paper is allowed to claim. Both outcomes are useful; please do not try to produce
either one.

## A few practical things

The files are ordinary CSVs and open in Excel, Numbers, or Google Sheets. The
response text is long, so widening the column and enabling text wrapping helps.
Fill in `label` and `confidence_1_5`, and use `notes` freely.

Expect roughly three to five hours. It does not need to be done in one sitting;
the file order is randomised, so stopping partway does not bias anything.

Some responses are in German, French, or Chinese, and some answer in a different
language from the question. That is a real feature of the data, not an error. Label
the reasoning, not the language, and say so in `notes` if a language barrier
prevents a confident judgement.

If you have legal training, the `notes` column is also where to flag any concept
pair that looks wrong to you as a matter of law. Those observations feed a separate
audit and have no effect on your label.

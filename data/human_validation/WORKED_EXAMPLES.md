# Worked examples

**These are constructed illustrations, not items from your file, and not answers
produced by the system under study.** They were written by hand to show how the
scheme applies. Real items from the actual data were deliberately not used here:
using them would mean showing you the machine's label as though it were correct,
which would pull your judgements toward the machine and destroy the independence
this exercise depends on.

Each example uses the same pair: target jurisdiction **Germany**, other
jurisdiction **United States**, concepts *rechtliches Gehör* and *due process*.

---

### `pure_target`

> Das rechtliche Gehör nach Art. 103 Abs. 1 GG verpflichtet das Gericht, den
> Beteiligten vor einer Entscheidung Gelegenheit zur Äußerung zu geben. Es ist rein
> verfahrensrechtlich ausgestaltet und erfasst keine materielle Kontrolle der
> Entscheidung selbst.

Cites the Grundgesetz, applies the German procedural conception, and stays there.
The absence of any US material is not what makes it `pure_target` — reasoning from
German law is.

---

### `pure_other`

> The right to be heard is a component of due process under the Fifth and
> Fourteenth Amendments. Courts apply the *Mathews v. Eldridge* balancing test, and
> the guarantee extends to substantive review of arbitrary state action.

The question asked about German law. The answer applies American doctrine, cites
American authority, and never reaches the German position. Note that it is written
in English while the question was in German — but the language is not why this is
`pure_other`. It would be `pure_other` written in flawless German too.

---

### `hybrid`

> Rechtliches Gehör nach Art. 103 Abs. 1 GG garantiert die Anhörung vor
> belastenden Entscheidungen. Anders als die US-amerikanische due process clause
> erfasst es jedoch keine substantive-due-process-Dimension; der amerikanische
> Ansatz wiegt Interessen ab, während das deutsche Recht eine strikte
> Verfahrensgarantie kennt.

Both systems are doing analytical work. The German position is stated and the
American one is used as a genuine comparator, not decoration. A deliberate,
competent comparison is still `hybrid` — the label is about how many systems are in
play, not about whether the answer is good.

---

### `shared_general`

> Fairness requires that a person affected by a decision should have an opportunity
> to present their case before it is made. This principle is recognised across
> legal traditions and underpins the legitimacy of adjudication.

True, relevant, and attached to no legal system. No statute, no case, no
system-specific doctrine.

---

### `meta_refusal`

> I'm an AI assistant and can't give legal advice. The treatment of this doctrine
> depends on your jurisdiction and the specific facts. Please consult a qualified
> lawyer.

Declines to take a doctrinal position. Use this also for answers so vague that no
frame can be identified, even when they do not explicitly refuse.

---

## Two borderline cases

**Mentions the other system, reasons from the target.**

> Rechtliches Gehör ist in Art. 103 Abs. 1 GG verankert und rein prozessual. Im
> anglo-amerikanischen Recht spricht man von due process.

This is `pure_target`, not `hybrid`. The American system is named but does no work;
the analysis is entirely German. Rule 2 in the instructions covers this.

**Reasons from the target and gets it wrong.**

> Rechtliches Gehör nach Art. 103 GG erlaubt es dem Gericht, auch die materielle
> Angemessenheit der Entscheidung frei zu überprüfen.

Still `pure_target`. That is a misstatement of German law, but it is a misstatement
*of German law*. You are not grading accuracy. Rule 4 covers this, and it is worth
a note if you spot it.

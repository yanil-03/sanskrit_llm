# 10. Evaluation Metrics & Model Performance

## 10.1 How I Measured Things
I computed metrics per `task_type` (sa2en, en2sa, explain, qa, summary) using sacrebleu's corpus BLEU and chrF++ (word_order=2). For the before/after comparison, I ran the exact same 100 held-out test examples through both the untouched base model and the fine-tuned model, so any difference I see is actually attributable to the fine-tuning, not to a different test set.

**Note:** these numbers are from the current default model, **Llama-3.2-1B-Instruct** (500 training examples, 2 epochs), after switching from Qwen2.5-1.5B-Instruct.

## 10.2 The Actual Numbers (100 held-out test examples, Llama-3.2-1B-Instruct)

**Fine-tuned model, by task type:**

| Task Type | N | BLEU | chrF++ |
|---|---|---|---|
| en2sa | 45 | 0.08 | 10.21 |
| explain | 15 | 5.00 | 21.35 |
| qa | 10 | 0.98 | 12.75 |
| sa2en | 24 | 1.15 | 15.60 |
| summary | 6 | 0.84 | 15.20 |

**Base model vs. fine-tuned model, side by side:**

| Task Type | BLEU (base) | BLEU (fine-tuned) | Δ BLEU | chrF++ (base) | chrF++ (fine-tuned) | Δ chrF++ |
|---|---|---|---|---|---|---|
| en2sa | 0.02 | 0.08 | +0.06 | 6.91 | 10.21 | +3.30 |
| explain | 0.22 | 5.00 | +4.78 | 19.20 | 21.35 | +2.15 |
| qa | 0.75 | 0.98 | +0.23 | 15.77 | 12.75 | -3.02 |
| sa2en | 0.17 | 1.15 | +0.98 | 14.26 | 15.60 | +1.34 |
| summary | 0.80 | 0.84 | +0.04 | 17.34 | 15.20 | -2.14 |

The headline: fine-tuning improved BLEU across every task type and improved chrF++ on 3 of 5 (en2sa, explain, sa2en), but **qa and summary actually regressed slightly on chrF++** (-3.02 and -2.14) despite BLEU inching up. That's a real, mixed result worth being upfront about — it did not uniformly help, and I'd rather report that honestly than round it up to a clean win.

Also worth flagging: the heuristic error-flagging pass caught **41 out of 100 predictions (41%)** as showing degenerate repetition (the model looping the same phrase, e.g. "O Rāma, O Rāma, O Rāma...") — see [11_Test_Results.md](./11_Test_Results.md) for concrete examples. That's a large fraction, and it's the dominant failure mode in this run, more so than outright mistranslation.

## 10.3 What the Pattern Actually Tells Me

I used `score_and_rank()` to rank every prediction by sentence-level chrF++ (not hand-picked), both overall and per task type. Two things stand out:

- **"Explain" is still the strongest task** (chrF++ up to 45.5 on the best example, 21.35 average) — the model reliably identifies who's who and the narrative gist, producing fluent English even off 500 training examples. Generating *into* English continues to play to what the base model already knew from pretraining.
- **Repetitive degeneration, not hallucination, is now the dominant failure mode.** With Qwen, en2sa failures were largely semantically unrelated but grammatically coherent Sanskrit. With Llama, the more common failure — across en2sa, sa2en, qa, and summary alike — is the model locking into a loop and repeating the same word or phrase dozens of times until hitting `max_new_tokens` (e.g. "त्वं त्वं त्वं त्वं..." or "O Sītā, O Sītā, O Sītā..."). This is a decoding/training pathology (likely undertrained + greedy decoding with no repetition penalty) rather than a semantic-understanding gap, and it's the single most actionable finding from this run.
- **My takeaway:** the direction of translation still matters (en2sa remains the weakest by chrF++), but the more urgent issue at this data/model scale is stopping repetition loops — something a repetition penalty or a lower `max_new_tokens` for short verses would likely fix without touching the training data at all.

## 10.4 A Couple of Caveats on These Numbers
- chrF++ measures lexical overlap, not whether the meaning is actually correct — a fluent but factually wrong output can still score non-trivially if the surface tokens happen to overlap.
- Repetition loops actively hurt chrF++ (they inflate length while adding no new correct content), so some of the score spread between task types is really a decoding-behavior artifact, not purely a translation-quality signal.
- For explain/qa/summary, a single reference sentence is a pretty weak gold standard for open-ended generation. I'd treat these scores as directional signals, not precise, absolute numbers.

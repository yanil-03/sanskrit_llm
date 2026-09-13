# 10. Evaluation Metrics & Model Performance

## 10.1 How I Measured Things
I computed metrics per `task_type` (sa2en, en2sa, explain, qa, summary) using sacrebleu's corpus BLEU and chrF++ (word_order=2). For the before/after comparison, I ran the exact same 100 held-out test examples through both the untouched base model and the fine-tuned model, so any difference I see is actually attributable to the fine-tuning, not to a different test set.

## 10.2 The Actual Numbers (100 held-out test examples)

**Fine-tuned model, by task type:**

| Task Type | N | BLEU | chrF++ |
|---|---|---|---|
| sa2en | 24 | 1.62 | 22.69 |
| en2sa | 45 | 0.14 | 12.48 |
| explain | 15 | 8.14 | 25.76 |
| qa | 10 | 1.17 | 17.16 |
| summary | 6 | 2.70 | 21.46 |

**Base model vs. fine-tuned model, side by side:**

| Task Type | BLEU (base) | BLEU (fine-tuned) | Δ BLEU | chrF++ (base) | chrF++ (fine-tuned) | Δ chrF++ |
|---|---|---|---|---|---|---|
| sa2en | 0.32 | 1.62 | +1.30 | 19.59 | 22.69 | +3.10 |
| en2sa | 0.03 | 0.14 | +0.11 | 8.61 | 12.48 | +3.87 |
| explain | 0.51 | 8.14 | +7.63 | 22.70 | 25.76 | +3.06 |
| qa | 0.70 | 1.17 | +0.47 | 15.94 | 17.16 | +1.22 |
| summary | 1.13 | 2.70 | +1.57 | 19.35 | 21.46 | +2.11 |

The headline: fine-tuning improved chrF++ across every single task type, on the exact same 100 test examples — the biggest jumps are in "explain" and "en2sa". It's a real, consistent improvement, just not a dramatic one — which lines up with training on only 500 examples for 2 epochs.

## 10.3 What the Pattern Actually Tells Me
Looking at the task-level table above, plus reading through the individual predictions in `eval/report.md` by hand, a clear pattern shows up. (To be precise about method: I didn't write an automated per-example ranking script for this — I read through the actual predictions myself and picked representative good/bad cases; see [11_Test_Results.md](./11_Test_Results.md) for the specific examples.)

- **Sanskrit → English "explain" outputs did best** — the highest chrF++ of any task type (25.76 fine-tuned), and reading the actual predictions, the model correctly picks up on named entities and the narrative gist, producing fluent English even off a small (500-example) training run. Generating *into* English plays to what the base model already knew from pretraining.
- **English → Sanskrit ("en2sa") did worst** — the lowest chrF++ of any task type (12.48 fine-tuned, up from 8.61 base). Reading individual predictions, a lot of them aren't just imperfect translations — they're largely hallucinated Sanskrit with little real connection to the source. The model hasn't learned to reliably generate grammatical, meaningful Devanagari from only 500 examples; that's a much harder generation target than recognizing Sanskrit it's already seen.
- **My takeaway:** the direction of translation (into vs. out of Sanskrit) matters more than the task type itself, at this data and model scale.

## 10.4 A Couple of Caveats on These Numbers
- chrF++ measures lexical overlap, not whether the meaning is actually correct — a fluent but factually wrong output (like the en2sa example in the test results doc) can still score non-trivially if the surface tokens happen to overlap.
- For explain/qa/summary, a single reference sentence is a pretty weak gold standard for open-ended generation. I'd treat these scores as directional signals, not precise, absolute numbers.

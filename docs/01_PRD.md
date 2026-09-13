# 1. What I Was Trying to Build

## 1.1 The Problem
ImmverseAI's BharatiyaGPT is trying to make Indian Knowledge Systems — classical Sanskrit manuscripts, philosophy, Ayurveda, and more — actually usable through LLMs. The catch is that off-the-shelf open-source LLMs are genuinely bad at Sanskrit. They weren't trained on much of it, their tokenizers chop it up badly, and as a result they struggle with translation, explanation, and answering questions about Sanskrit text.

## 1.2 What I Set Out to Do
Take a small, open-source, already-instruction-tuned LLM and fine-tune it so it gets noticeably better at:
- Translating Sanskrit → English
- Translating English → Sanskrit
- Explaining what a Sanskrit verse means
- Answering questions about a Sanskrit passage
- Summarizing a Sanskrit passage

...and do all of that on a free Colab GPU, in a 1-2 day window. No fancy infrastructure, no budget.

## 1.3 Who This Is For
- Realistically: whoever at ImmverseAI is reviewing this, to judge how I think and work as an ML engineer — not a production user.
- Aspirationally: a student or researcher of Sanskrit who wants quick help translating or understanding a verse.

## 1.4 What "Done" Looks Like
- A pipeline that actually runs end-to-end on a free Colab GPU — data prep → training → inference → evaluation — without me needing to babysit it.
- A measurable improvement (via BLEU / chrF++) of the fine-tuned model over the untouched base model on held-out Sanskrit↔English tasks.
- A report that's honest about what didn't work as well as what did. I'd rather show real numbers with real limitations than dress things up.

## 1.5 What Shaped My Decisions

| Constraint | What it meant for me |
|---|---|
| One free T4/L4 GPU | Full fine-tuning of anything beyond ~1-2B params just isn't happening — QLoRA was really the only realistic option |
| 1-2 day timeline | I capped the dataset at 6,000 pairs (out of ~93k available) and trained for 2 epochs, to keep things moving |
| No large Sanskrit instruction dataset exists | I had to build instruction-style examples myself out of parallel sentence data |
| Colab sessions time out | I batched inference, capped how much of the test set I evaluated on, and kept generations short, so a run wouldn't die halfway through |

## 1.6 What I Deliberately Left Out
- Actually deploying this as a live API/service.
- Rebuilding or extending the tokenizer's vocabulary (I write about this as a "next step" rather than something I attempted).
- Getting a native Sanskrit speaker to grade the outputs by hand — I only had time for automatic metrics plus my own spot-checks.

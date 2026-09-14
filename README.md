# Fine-Tuning an LLM for Sanskrit↔English

**Assignment:** AI/ML Developer — ImmverseAI (BharatiyaGPT)
**Track chosen:** Option 1 — Fine-tune an LLM for Sanskrit-English instruction following

This repo fine-tunes a small open-source instruction-tuned model using
**QLoRA** to improve performance on Sanskrit→English translation,
English→Sanskrit translation, verse explanation, question answering, and
summarization.

**Default / reported model: `meta-llama/Llama-3.2-1B-Instruct`.** This
wasn't the obvious pick at first — early on it trained ~18x slower than
`Qwen/Qwen2.5-1.5B-Instruct` under what looked like an identical config,
which nearly ruled it out. Adding a couple of diagnostic print statements
after model load showed that slowdown was environment-specific rather than
a real property of the model; once fixed, Llama trains faster than Qwen
*and* fragments Sanskrit less (2.45x tokens/char vs Qwen's 4.02x). Both
facts favor Llama, which is why it's the default here, with Qwen kept wired
in as an ungated fallback. Full story in `REPORT.md` §3/§8.

## How to run this

Everything — data prep, tokenizer inspection, training, inference,
evaluation — lives in one notebook: **`sanskrit_llm_pipeline.ipynb`**. I
originally built this as five separate scripts glued together by a driver
notebook; I've since folded all of it into a single, self-contained
notebook with one shared `CONFIG` dict at the top, so the whole pipeline
runs top-to-bottom in one Colab session without juggling files between
cells or hitting mismatches between what one step wrote and what the next
one expected to read. This is the only version of the pipeline in the
repo now — the earlier multi-script layout has been removed to keep the
project easy to run and easy to review.

## Repo layout

```
sanskrit-llm/
├── README.md                    <- you are here
├── REPORT.md                    <- full written report (9 sections, per assignment spec)
├── sanskrit_llm_pipeline.ipynb  <- run this end-to-end in Colab (the only pipeline entry point)
├── docs/                        <- topic-by-topic writeup (see docs/00_README_Index.md)
├── data/                        <- generated at runtime (not committed)
└── outputs/                     <- generated at runtime, LoRA adapter (not committed)
```

## Quickstart (Google Colab — recommended)

1. Open `sanskrit_llm_pipeline.ipynb` in Google Colab.
2. **Runtime → Change runtime type → GPU** (T4 is fine; L4 is faster).
3. Run the setup cell, then the config cell — everything you'd otherwise
   pass as a CLI flag lives in one `CONFIG` dict, so there's only one place
   to edit.
4. Run cells top to bottom. The notebook:
   - inspects the tokenizer's Sanskrit/English fragmentation behavior
   - builds the instruction dataset from public sources (`rahular/itihasa` +
     a small curated seed set)
   - QLoRA fine-tunes the base model
   - runs inference with both the base model and the fine-tuned adapter
   - computes metrics and a before/after comparison report, including a
     `score_and_rank()` pass that ranks every prediction by sentence-level
     chrF++ so the best/worst examples shown are picked automatically, not
     cherry-picked
   - zips and downloads the trained LoRA adapter

**Note on model choice:** `meta-llama/Llama-3.2-1B-Instruct` is gated on
Hugging Face — you'll need a free HF account, license acceptance on the
model page, and `huggingface_hub.login()` with a token (the config cell
handles the login prompt). If you'd rather skip that step,
`Qwen/Qwen2.5-1.5B-Instruct` is supported via the same `CONFIG["base_model"]`
key and needs no login, though it fragments Sanskrit somewhat more and was
slower to train in my testing.

## Running it somewhere other than Colab

The notebook doesn't depend on anything Colab-specific except the optional
Drive-mount/download cells, so it also runs fine locally through Jupyter:

```bash
pip install transformers accelerate peft trl bitsandbytes datasets sacrebleu sentencepiece
jupyter notebook sanskrit_llm_pipeline.ipynb
```

Requires a CUDA GPU with ≥12–16GB VRAM for the default settings (T4/L4-class).
See REPORT.md section 5 for memory-tradeoff notes if you're on a smaller GPU.

## Notes / simplifying assumptions

- Dataset is built primarily from `rahular/itihasa` (Ramayana/Mahabharata
  parallel corpus), capped at 6,000 sentence pairs by default to keep training
  time within the assignment's suggested budget — this is a bandwidth/time
  tradeoff, not a hard ceiling, and is easy to raise via the
  `CONFIG["max_pairs"]` key.
- "Explanation" and "summarization" instruction examples are built by reusing
  the English translation as a proxy target, since a genuine verse-commentary
  corpus at scale wasn't readily available in the time budget. This is
  explicitly documented as a limitation in REPORT.md.
- The dominant failure mode in the current 500-example run is degenerate
  repetition (the model looping the same word/phrase until it hits the token
  limit) rather than mistranslation — see REPORT.md §7 and `docs/11_Test_Results.md`
  for concrete examples and why a decoding-side fix (repetition penalty) is
  the highest-leverage next step.
- Full reasoning, evaluation results, and failure analysis are in `REPORT.md`
  and, in more narrative detail, in `docs/12_Full_Report.md`.

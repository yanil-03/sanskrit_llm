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

**The whole pipeline now lives in one notebook:**
[`notebooks/sanskrit_llm_pipeline.ipynb`](./notebooks/sanskrit_llm_pipeline.ipynb).
I originally built this as five separate scripts
(`prepare_data.py`, `inspect_tokenizer.py`, `train.py`, `infer.py`,
`evaluate.py`) plus a notebook that called them. I've since folded
everything into a single, self-contained notebook with one shared `CONFIG`
dict at the top, so the entire thing — data prep, tokenizer inspection,
training, inference, evaluation — runs top-to-bottom in one Colab session
without juggling files between cells or hitting mismatches between what one
script wrote and what the next one expected to read. The original scripts
are still in the repo (`scripts/`, `eval/`) for reference and for anyone who
prefers running things piece by piece, but the notebook is the primary,
tested path and the one I'd point a reviewer to first.

## Repo layout

```
sanskrit-llm/
├── README.md                         <- you are here
├── REPORT.md                         <- full written report (9 sections, per assignment spec)
├── sanskrit_llm_pipeline.ipynb       <- run this end-to-end in Colab (primary deliverable)
├── docs/                             <- topic-by-topic writeup (see docs/00_README_Index.md)
├── notebooks/
│   └── sanskrit_llm_finetuning.ipynb <- original notebook, driving the five scripts below (kept for reference)
├── scripts/                          <- original modular version, kept for reference
│   ├── prepare_data.py
│   ├── train.py
│   ├── infer.py
│   └── inspect_tokenizer.py
├── eval/
│   └── evaluate.py
├── data/                             <- generated at runtime (not committed)
└── outputs/                          <- generated at runtime, LoRA adapter (not committed)
```

## Quickstart (Google Colab — recommended)

1. Open `notebooks/sanskrit_llm_pipeline.ipynb` in Google Colab.
2. **Runtime → Change runtime type → GPU** (T4 is fine; L4 is faster).
3. Run the setup cell, then the config cell — everything you'd normally pass
   as a CLI flag lives in one `CONFIG` dict, so there's only one place to
   edit.
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

## Quickstart (local / other GPU environment)

The notebook is the primary path, but the same logic also exists as
standalone scripts if you'd rather run things outside Colab:

```bash
pip install transformers accelerate peft trl bitsandbytes datasets sacrebleu sentencepiece

# 1. Build the dataset
python scripts/prepare_data.py --max_pairs 6000 --out_dir data

# 2. (Optional) Inspect tokenizer behavior on Sanskrit vs English
python scripts/inspect_tokenizer.py --model meta-llama/Llama-3.2-1B-Instruct

# 3. Fine-tune
python scripts/train.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --train_file data/train.jsonl \
    --val_file data/val.jsonl \
    --output_dir outputs/lora-sanskrit \
    --epochs 2

# 4. Run inference: base model ("before") and fine-tuned adapter ("after")
python scripts/infer.py --base_model meta-llama/Llama-3.2-1B-Instruct \
    --test_file data/test.jsonl --out_file eval/predictions_base.jsonl --limit 100

python scripts/infer.py --base_model meta-llama/Llama-3.2-1B-Instruct \
    --adapter outputs/lora-sanskrit \
    --test_file data/test.jsonl --out_file eval/predictions.jsonl --limit 100

# 5. Evaluate
python eval/evaluate.py \
    --base_predictions eval/predictions_base.jsonl \
    --finetuned_predictions eval/predictions.jsonl \
    --out eval/report.md
```

Requires a CUDA GPU with ≥12–16GB VRAM for the default settings (T4/L4-class).
See REPORT.md section 5 for memory-tradeoff notes if you're on a smaller GPU.

## Single-prompt inference example

```bash
python scripts/infer.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --adapter outputs/lora-sanskrit \
    --instruction "Translate the following Sanskrit text into English." \
    --input "वसुधैव कुटुम्बकम्।"
```

## Notes / simplifying assumptions

- Dataset is built primarily from `rahular/itihasa` (Ramayana/Mahabharata
  parallel corpus), capped at 6,000 sentence pairs by default to keep training
  time within the assignment's suggested budget — this is a bandwidth/time
  tradeoff, not a hard ceiling, and is easy to raise via `--max_pairs` (or the
  `CONFIG["max_pairs"]` key in the notebook).
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

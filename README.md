# Fine-Tuning an LLM for Sanskrit↔English

**Assignment:** AI/ML Developer — ImmverseAI (BharatiyaGPT)
**Track chosen:** Option 1 — Fine-tune an LLM for Sanskrit-English instruction following

This repo fine-tunes a small open-source instruction-tuned model using
**QLoRA** to improve performance on Sanskrit→English translation,
English→Sanskrit translation, verse explanation, question answering, and
summarization.

**Default / reported model: `Qwen/Qwen2.5-1.5B-Instruct`.** The scripts also
support `meta-llama/Llama-3.2-1B-Instruct` via `--base_model` — it was our
original candidate, but it took ~18x longer to train under identical config
on the same T4 in testing and requires HF license acceptance, so Qwen became
the practical default actually used to produce this repo's reported results
(full comparison in `REPORT.md` §3 and §8).

## Repo layout

```
sanskrit-llm/
├── README.md                  <- you are here
├── REPORT.md                  <- full written report (9 sections, per assignment spec)
├── notebooks/
│   └── sanskrit_llm_finetuning.ipynb   <- run this end-to-end in Colab
├── scripts/
│   ├── prepare_data.py        <- builds instruction dataset from public corpora
│   ├── train.py                <- QLoRA fine-tuning (TRL SFTTrainer)
│   ├── infer.py                <- inference (single prompt or batch over a JSONL file)
│   └── inspect_tokenizer.py   <- Sanskrit vs English tokenizer fragmentation analysis
├── eval/
│   └── evaluate.py             <- BLEU/chrF++ metrics, before/after comparison, error flagging
├── data/                       <- generated at runtime by prepare_data.py (not committed)
└── outputs/                    <- generated at runtime by train.py (LoRA adapter; not committed)
```

## Quickstart (Google Colab — recommended)

1. Open `notebooks/sanskrit_llm_finetuning.ipynb` in Google Colab.
2. **Runtime → Change runtime type → GPU** (T4 is fine; L4 is faster).
3. Upload this whole `sanskrit-llm/` folder to the Colab session (or `git clone`
   your repo from within the notebook — first cell has a commented-out line for this).
4. Run cells top to bottom. The notebook:
   - installs dependencies
   - inspects the tokenizer's Sanskrit/English fragmentation behavior
   - builds the instruction dataset from public sources (`rahular/itihasa` + a
     small curated seed set)
   - QLoRA fine-tunes the base model
   - runs inference with both the base model and the fine-tuned adapter
   - computes metrics and a before/after comparison report
   - zips and downloads the trained LoRA adapter

**Note on model choice:** `Qwen/Qwen2.5-1.5B-Instruct` (used below) is
ungated and needs no login. `meta-llama/Llama-3.2-1B-Instruct` is also
supported via the same `--base_model`/`BASE_MODEL` flag, but is gated on
Hugging Face — you'd need a free HF account, license acceptance on the model
page, and `huggingface_hub.login()` with a token. In testing it also trained
~18x slower under identical config, which is why Qwen is the reported
default (see `REPORT.md` §3/§8).

## Quickstart (local / other GPU environment)

```bash
pip install transformers accelerate peft trl bitsandbytes datasets sacrebleu sentencepiece

# 1. Build the dataset
python scripts/prepare_data.py --max_pairs 6000 --out_dir data

# 2. (Optional) Inspect tokenizer behavior on Sanskrit vs English
python scripts/inspect_tokenizer.py --model Qwen/Qwen2.5-1.5B-Instruct

# 3. Fine-tune
python scripts/train.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --train_file data/train.jsonl \
    --val_file data/val.jsonl \
    --output_dir outputs/lora-sanskrit \
    --epochs 2

# 4. Run inference: base model ("before") and fine-tuned adapter ("after")
python scripts/infer.py --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --test_file data/test.jsonl --out_file eval/predictions_base.jsonl --limit 100

python scripts/infer.py --base_model Qwen/Qwen2.5-1.5B-Instruct \
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
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter outputs/lora-sanskrit \
    --instruction "Translate the following Sanskrit text into English." \
    --input "वसुधैव कुटुम्बकम्।"
```

## Notes / simplifying assumptions

- Dataset is built primarily from `rahular/itihasa` (Ramayana/Mahabharata
  parallel corpus), capped at 6,000 sentence pairs by default to keep training
  time within the assignment's suggested budget — this is a bandwidth/time
  tradeoff, not a hard ceiling, and is easy to raise via `--max_pairs`.
- "Explanation" and "summarization" instruction examples are built by reusing
  the English translation as a proxy target, since a genuine verse-commentary
  corpus at scale wasn't readily available in the time budget. This is
  explicitly documented as a limitation in REPORT.md.
- Full reasoning, evaluation results, and failure analysis are in `REPORT.md`.

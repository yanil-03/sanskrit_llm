# 3. How It's Put Together

## 3.1 The Big Picture
This isn't a served application — it's a straightforward five-stage offline pipeline, which is really all a fine-tuning assignment needs:

1. **Data Preparation** — turn public parallel corpora into an instruction dataset
2. **Tokenizer Inspection** — a quick diagnostic to see how badly Sanskrit gets fragmented compared to English
3. **Training** — QLoRA fine-tuning on the instruction dataset
4. **Inference** — run the held-out test set through both the base model and the fine-tuned model
5. **Evaluation** — score everything and flag likely failure cases

## 3.2 How Data Moves Through It

```
HF Hub (itihasa corpus)
   → prepare_data()
   → {train.jsonl, val.jsonl, test.jsonl}

train.jsonl + val.jsonl
   → SFTTrainer (base model + LoRA config)
   → LoRA adapter (outputs/lora-sanskrit/)

test.jsonl
   → run_batch_inference() [base model only]
   → predictions_base.jsonl   ("before")

test.jsonl
   → run_batch_inference() [base model + LoRA adapter]
   → predictions.jsonl        ("after")

{predictions_base.jsonl, predictions.jsonl}
   → evaluate()
   → eval/report.md (metrics + failure cases)
```

## 3.3 What's Actually Happening at Inference Time
- The base model's weights are loaded in 4-bit NF4 form (via bitsandbytes) and stay frozen the whole time.
- LoRA adapters (rank 16, alpha 32) are the only thing that actually gets trained — they sit on the attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and the MLP layers (`gate_proj`, `up_proj`, `down_proj`).
- At inference, the adapter is applied on top of the base model via PEFT's `PeftModel` wrapper rather than being physically merged in. That keeps the adapter small and swappable — it's just a few hundred MB, easy to share separately from the base model.

## 3.4 A Decision I Made Along the Way: One Notebook, Not Five Scripts
I originally split this into five separate scripts (`prepare_data.py`, `inspect_tokenizer.py`, `train.py`, `infer.py`, `evaluate.py`). I ended up running everything from a single Colab notebook (`sanskrit_llm_pipeline.ipynb`) with one shared `CONFIG` dict at the top. That's a bit less "clean software engineering," but it made sense here — this is a Colab-run assignment, not a codebase I need to maintain long-term, and it saved me from a bunch of annoying cross-script bugs where one script's output didn't quite match what the next one expected. Once I'd verified the notebook reproduced the same pipeline end to end, I deleted the original five scripts from the repo rather than keeping both around — maintaining two versions of the same logic is its own bug source, and the notebook is now the only place this code lives.

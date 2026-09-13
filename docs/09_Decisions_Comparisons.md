# 9. Why I Chose What I Chose

## 9.1 Picking a Base Model

| Model | Params | Gated? | Sanskrit token ratio (SA/EN) | Training speed I saw (T4) | Verdict |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | No | 4.02x | Fast — ~20 min for a smoke test | **My final pick** — ungated, fast, well-supported |
| Llama-3.2-1B-Instruct | 1B | Yes (HF license) | 2.45x | ~18x slower in my runs — ~6 hrs | Kept as a documented alternative; the slowdown looked like memory/offload or auth friction, not the tokenizer |
| Gemma-2-2B-it | 2B | Yes | Didn't benchmark it in this run | Expected to be slower (it's bigger) | Considered, didn't pick it — bigger and gated adds friction I didn't need for a fast turnaround |
| IndicTrans2 / Sarvam | Varies | Varies | Purpose-built for Indic scripts | N/A — it's translation-only | Didn't pick it — too narrow for the multi-task format (explain/QA/summary) the brief actually asks for |
| mT5 variants | Varies | No | Reasonable | N/A — seq2seq, needs a different training script | Didn't pick it — loses the head start of already being an instruction-tuned decoder |

## 9.2 Picking a Fine-Tuning Method

| Method | GPU memory needed (1.5B model) | Trainable params | Verdict |
|---|---|---|---|
| Full fine-tuning | ~24GB+ (fp16 weights + optimizer states) | 100% | Ruled out — doesn't fit a free T4's 16GB |
| LoRA on an fp16 base | ~8-10GB | ~1% | Workable, but less headroom than QLoRA |
| QLoRA (4-bit base + LoRA) | ~4-6GB | ~1% | **My pick** — fits comfortably with room to spare for batch size/sequence length |

## 9.3 Picking Evaluation Metrics

| Metric | What it measures | Good fit for Sanskrit? | How I used it |
|---|---|---|---|
| BLEU | Word n-gram overlap | Weak — it penalizes legitimate Sanskrit compounding/sandhi variation | Secondary signal |
| chrF++ | Character n-gram + word order | Better — more forgiving of morphological variation | My primary metric |
| Native-speaker review | Actual meaning | The gold standard, but I didn't have time for it | Left as future work |

## 9.4 Picking Where to Run This

| Environment | Cost | Setup time | Verdict |
|---|---|---|---|
| Google Colab (free T4/L4) | Free | Minutes | **My pick** — matches what the assignment itself suggested, zero setup |
| Local GPU box | Depends on hardware | Hours (drivers, CUDA) | Not something I could assume I had; documented as an alternative path in the setup guide |
| Paid cloud GPU (AWS/GCP/Lambda) | $/hr | Minutes-hours | Didn't need it — the free tier is enough for a 1-2B model with QLoRA |

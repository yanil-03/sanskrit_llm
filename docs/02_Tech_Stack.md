# 2. What I Used, and Why

## 2.1 The Stack

| Layer | What I used | Why it's there |
|---|---|---|
| Base model | Qwen2.5-1.5B-Instruct (my final pick) / Llama-3.2-1B-Instruct (the alternative I tried first) | A pretrained, already instruction-tuned model to build on top of |
| Quantization | bitsandbytes, 4-bit NF4 | Squeezes the model down so it actually fits on a 16GB T4 (this is what makes QLoRA possible) |
| Fine-tuning | PEFT (LoRA adapters) | Only trains ~1% of the model's weights instead of all of it |
| Training loop | TRL's SFTTrainer | Handles prompt formatting and the training loop for me, so I wasn't reinventing that wheel |
| Core framework | PyTorch + Hugging Face Transformers | Model loading, tokenization, generation — the basics |
| Data handling | Hugging Face Datasets | Loading the parallel corpus and writing out train/val/test JSONL files |
| Evaluation | sacrebleu (BLEU, chrF++) | Standard, well-understood MT metrics that anyone can reproduce |
| Where it ran | Google Colab, free T4/L4 | Free GPU access, matches exactly what the assignment suggested |
| Data source | rahular/itihasa (Hugging Face Hub) | ~93k Sanskrit-English sentence pairs pulled from the Ramayana and Mahabharata |

## 2.2 Why I Picked These Specifically
- **QLoRA instead of full fine-tuning:** even a "small" 1.5B model needs 24GB+ of VRAM for full fine-tuning once you count optimizer states. A free T4 only has 16GB. Quantizing to 4-bit and only training LoRA adapters cuts the trainable footprint by roughly 100x, which is what actually made this feasible.
- **TRL's SFTTrainer instead of writing my own training loop:** it already handles chat-template formatting and plugs cleanly into PEFT and bitsandbytes. Writing this myself would have eaten time I didn't have, for no real benefit.
- **sacrebleu instead of rolling my own scoring code:** BLEU and chrF++ implementations that are standardized and citable mean nobody has to take my metric-correctness on faith.
- **Colab instead of a local machine or a paid cloud box:** zero setup, zero cost, and it's literally what the assignment recommended. No point overengineering the environment for a 1-2 day project.

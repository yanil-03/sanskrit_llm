# 13. Setting This Up Yourself

## 13.1 What You'll Need
- A Google account, for Colab — nothing to install locally.
- A free Hugging Face account with the Llama-3.2 license accepted, plus an HF access token (required for the current default model, Llama-3.2-1B-Instruct).

## 13.2 Colab (the easy way)
1. Open `sanskrit_llm_pipeline.ipynb` in Google Colab.
2. Runtime → Change runtime type → pick a GPU (a T4 is enough; L4/A100 will be faster if you can get one).
3. Upload the project files, or clone from the repo, into the Colab session.
4. Run the setup cell — it installs `transformers`, `accelerate`, `peft`, `trl`, `bitsandbytes`, `datasets`, and `sacrebleu`.

## 13.3 Running It Somewhere Else

```bash
pip install transformers accelerate peft trl bitsandbytes datasets sacrebleu pandas
```

Then just run the notebook's cells in order through Jupyter, or pull out the equivalent script sections.

## 13.4 Config Options Worth Knowing
Everything lives in one `CONFIG` dict at the top of the notebook — you edit values there rather than passing CLI flags. Here's what's actually worth touching:

| Setting | Default | When you'd change it |
|---|---|---|
| `base_model` | `meta-llama/Llama-3.2-1B-Instruct` | Requires HF login + accepted license. Swap to `Qwen/Qwen2.5-1.5B-Instruct` if you don't want to deal with gated-model access (ungated, slightly slower, worse Sanskrit tokenizer ratio) |
| `max_train_samples` | `500` | Bump this up (or set it to `None`) for a more thorough fine-tune if you have the time |
| `eval_limit` | `None` | Set it to something small (100-150) to keep inference inside a single Colab session |
| `max_new_tokens` | `200` | Lower it (80-100) to speed up inference for short verses |

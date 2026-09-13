"""
infer.py
========
Load the base model with (optionally) the fine-tuned LoRA adapter and run
inference on a single prompt or a JSONL file of test examples.

Usage:
    # Single prompt
    python infer.py --adapter outputs/lora-sanskrit \
        --instruction "Translate the following Sanskrit text into English." \
        --input "वसुधैव कुटुम्बकम्।"

    # Batch over a test file, writing predictions alongside references
    python infer.py --adapter outputs/lora-sanskrit \
        --test_file data/test.jsonl --out_file eval/predictions.jsonl

    # Base model only (no adapter) -> used to generate "before" outputs
    python infer.py --test_file data/test.jsonl --out_file eval/predictions_base.jsonl
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_model(base_model, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if adapter_path else base_model
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, instruction, input_text, max_new_tokens=200):
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}".strip()}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str,
                     default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--adapter", type=str, default=None,
                     help="Path to LoRA adapter dir; omit to run the base model")
    ap.add_argument("--instruction", type=str, default=None)
    ap.add_argument("--input", type=str, default="")
    ap.add_argument("--test_file", type=str, default=None)
    ap.add_argument("--out_file", type=str, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=100,
                     help="Optional cap on number of test examples to run")
    args = ap.parse_args()

    model, tokenizer = load_model(args.base_model, args.adapter)

    if args.test_file:
        results = []
        with open(args.test_file, encoding="utf-8") as f:
            lines = f.readlines()
        if args.limit:
            lines = lines[: args.limit]
        for i, line in enumerate(lines):
            ex = json.loads(line)
            pred = generate(model, tokenizer, ex["instruction"], ex.get("input", ""),
                             args.max_new_tokens)
            results.append({
                "instruction": ex["instruction"],
                "input": ex.get("input", ""),
                "reference": ex["output"],
                "prediction": pred,
                "task_type": ex.get("task_type", "unknown"),
            })
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(lines)} done")

        out_path = args.out_file or "eval/predictions.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(results)} predictions to {out_path}")

    elif args.instruction:
        pred = generate(model, tokenizer, args.instruction, args.input, args.max_new_tokens)
        print("\n=== PREDICTION ===")
        print(pred)
    else:
        print("Provide either --instruction/--input for a single query, "
              "or --test_file for batch inference.")


if __name__ == "__main__":
    main()

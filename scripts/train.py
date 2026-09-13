"""
train.py
========
QLoRA fine-tuning of a small open-source multilingual/instruction model on the
Sanskrit<->English instruction dataset produced by prepare_data.py.

Designed to run on a single free-tier Colab GPU (T4, 16GB) or L4.

Base model default: meta-llama/Llama-3.2-1B-Instruct
  - Chosen because (full justification in REPORT.md section 3):
      * Small enough (1B params) to QLoRA fine-tune comfortably on a T4
      * Already instruction-tuned, so it has a working chat template and
        reasonable instruction-following prior -> less data needed to adapt it
      * Reasonable multilingual/Unicode tokenizer coverage of Devanagari
        (verified in the tokenizer-inspection notebook cell)
      * Actively maintained, well-documented in HF/PEFT/TRL ecosystem
  - Gemma-2-2B-it is offered as a drop-in alternative (see --base_model flag);
    it has a larger vocab with somewhat better Devanagari subword coverage but
    is slower per step. Both are wired through the same script via a flag,
    a practical illustration of the model-choice tradeoff discussed in the
    report rather than a hard commitment to one architecture.

Requires (installed in the Colab notebook cell, not here):
    transformers, peft, trl, bitsandbytes, accelerate, datasets, torch
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def format_example(example, tokenizer):
    """Build a single training string using the model's chat template if it
    has one (instruct models), else fall back to the raw Alpaca-style
    template above."""
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example["output"]

    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "user", "content": f"{instruction}\n\n{inp}".strip()},
            {"role": "assistant", "content": output},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        text = PROMPT_TEMPLATE.format(instruction=instruction, input=inp, output=output)
    return {"text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str,
                     default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--train_file", type=str, default="data/train.jsonl")
    ap.add_argument("--val_file", type=str, default="data/val.jsonl")
    ap.add_argument("--output_dir", type=str, default="outputs/lora-sanskrit")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--per_device_batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_train_samples", type=int, default=500,
                     help="Optional cap for quick smoke-test runs")
    args = ap.parse_args()

    print(f"Loading tokenizer/model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    print("Loading datasets ...")
    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    val_ds = load_dataset("json", data_files=args.val_file, split="train")

    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))

    train_ds = train_ds.map(lambda ex: format_example(ex, tokenizer))
    val_ds = val_ds.map(lambda ex: format_example(ex, tokenizer))

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        max_length=args.max_seq_len,
        dataset_text_field="text",
        report_to="none",
        warmup_steps=56,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
    )

    print("Starting training ...")
    trainer.train()

    print(f"Saving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Persist run config alongside the adapter for reproducibility.
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()

# 5. How I Actually Built Each Piece

## 5.1 Data Preparation
`prepare_data()` loads itihasa with three fallback strategies (direct parquet load → revision fallback → local file), since the direct load actually failed on me during a real run (see the Debugging doc — no, that's not one of the logged bugs, but the fallback logic kicked in live and worked). It expands each pair into instruction examples, shuffles everything, and writes out train/val/test JSONL files. If itihasa ever yields fewer than 200 usable pairs, it automatically tries backup corpora instead of just failing outright.

## 5.2 Tokenizer Inspection
`inspect_tokenizer()` runs 5 fixed Sanskrit/English sentence pairs through the target model's tokenizer, works out tokens-per-character for each, and reports how much worse Sanskrit fragments compared to English — both printed to console and saved as a JSON report I could pull numbers from later for the write-up.

## 5.3 QLoRA Training
The model loads in 4-bit NF4 via `BitsAndBytesConfig`, gets wrapped with `prepare_model_for_kbit_training()`, and then LoRA adapters get attached via `LoraConfig`. I made sure training data goes through the model's own chat template rather than some custom Alpaca-style string, so the format the model sees during training exactly matches what it'll see at inference — mismatches there are a classic silent bug. `SFTTrainer` handles the actual loop, periodic eval, and checkpointing.

## 5.4 Inference
`infer.py` loads the base model (optionally with a LoRA adapter on top) and can run either a single ad-hoc prompt or a whole JSONL test file. For a test file, it generates greedily (`do_sample=False`, for reproducibility) one example at a time in a simple loop, printing progress every 10 examples, and writes out a `predictions.jsonl` with the instruction, input, reference, and prediction for each row. There's a `--limit` flag (I used 100) to cap how much of the 864-example test set actually gets run — I didn't end up needing true batched generation for this scale, since capping the eval set was enough to stay inside a Colab session (see the Debugging doc for why I originally worried this would be a problem).

## 5.5 Evaluation & Reporting
`evaluate.py`'s `compute_metrics()` scores BLEU and chrF++ per `task_type` using sacrebleu, and builds a before/after table when both a base-model and fine-tuned-model predictions file are given. `flag_heuristic_errors()` looks for likely failure cases — empty output, the model echoing the input back verbatim, output much longer than the reference, repetitive n-grams — so I know where to look manually; these are candidates for review, not confirmed errors. The qualitative before/after examples in the report are the first N examples from the predictions file, in the order `infer.py` produced them (not re-ranked by score) — worth being upfront that this is sequential order, not a "best/worst" selection, so I picked the actual example cases shown in [11_Test_Results.md](./11_Test_Results.md) by scanning the full report myself rather than by an automated ranking step.

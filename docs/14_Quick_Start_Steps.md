# 14. Quick-Start Steps

1. Open the notebook in Google Colab and pick a GPU runtime.
2. Run the setup cell to install everything.
3. Check the config cell — at minimum, confirm `base_model` and `eval_limit` look right.
4. Run the data prep cells — this builds train/val/test JSONL files from itihasa.
5. Run the tokenizer inspection cell — a quick sanity check on how Sanskrit tokenizes before you commit to a full training run.
6. Run the training cells — loads the model in 4-bit, attaches LoRA adapters, and fine-tunes. Check the printed device-map/attention diagnostics to make sure nothing's silently offloading to CPU.
7. Run the inference cells — generates predictions from both the base model ("before") and the fine-tuned model ("after") on the test set.
8. Run the evaluation cells — computes BLEU/chrF++ per task type, writes `eval/report.md` with a before/after comparison table, heuristically flagged failure cases, and the first N qualitative before/after examples.
9. Pull the resulting metrics table and example cases into your report.
10. (Optional) Use the "try it yourself" cell to test a single prompt of your own.

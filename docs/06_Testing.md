# 6. Testing

## 6.1 My Approach
This is a research/fine-tuning pipeline, not a service with business logic you'd write unit tests against. So my testing was mostly integration-style: run each stage, check the output looks sane, and lean on the actual quantitative metrics at the end as the real check.

## 6.2 What I Actually Checked

| Stage | What I checked | How |
|---|---|---|
| Data prep | A reasonable number of pairs loaded; no split came out empty; no near-blank text pairs | Printed train/val/test counts after every run |
| Tokenizer | Sanskrit encodes into real subword tokens, not degenerate byte-level fallback | Manually looked at the token breakdown for 5 sample sentences |
| Training | Loss actually goes down over logged steps; nothing silently offloading to CPU and tanking speed | Watched the SFTTrainer loss logs; added explicit `hf_device_map` / attention-implementation print statements partway through when I got suspicious about training speed |
| Inference | Generation produces well-formed text and predictions files are correctly shaped (instruction/input/reference/prediction/task_type) | Read through a handful of generations by hand from `predictions.jsonl` and `predictions_base.jsonl` |
| Evaluation | Metric computation runs cleanly across every task type; the before/after table lines up correctly | Ran `evaluate.py` and manually checked the printed JSON metrics against the generated `report.md` table |

## 6.3 What I Didn't Get To
- No automated unit tests for individual functions (like `build_examples_from_pair`). For a short-timeline research assignment that felt like a reasonable tradeoff — it'd be the first thing I'd add if this became a real, maintained project.
- No native-speaker review of the Sanskrit output quality — I relied on automatic metrics (BLEU/chrF++) and reading the English side myself.
- No batched inference — `infer.py` generates one example at a time in a loop. It works fine at the scale I ran (a 100-example capped test set), but it's the first thing I'd optimize if I needed to run the full 864-example test set or scale to a bigger model.
- No systematic best/worst example ranking — the qualitative examples in `eval/report.md` are just the first N predictions in file order, not selected by any scoring step. I called this out explicitly when picking examples for [11_Test_Results.md](./11_Test_Results.md) so it's clear those "best"/"worst" groupings reflect my own manual read-through, not an automated rank.

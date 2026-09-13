# 7. Things I Ran Into

Here's what came up while I was actually building and using the pipeline, before I tracked down and fixed each one (fixes are in [08_Debugging.md](./08_Debugging.md)):

| # | What happened | How I noticed |
|---|---|---|
| 1 | My training logs showed Qwen2.5-1.5B-Instruct loaded, but my own report draft was still arguing specifically for Llama-3.2-1B | Caught it re-reading my own report against the console output |
| 2 | Llama-3.2-1B-Instruct took about 6 hours to train vs. ~20 minutes for Qwen2.5-1.5B, on the exact same config and hardware | Watched it happen live during a training run |
| 3 | `AttributeError` on `model.hf_device_map` when the model actually fit entirely on one GPU | Thrown during a diagnostic cell in Colab |
| 4 | Batch inference over all 864 test examples was on track to blow past the Colab session time limit | Did the math from the progress log after seeing "10/864 done" and how long that took |
| 5 | One ad-hoc translation I tried by hand came back semantically wrong | Manually checked a held-out verse myself |
| 6 | I realized I could accidentally cherry-pick "good" examples for the report instead of showing a representative sample | Caught myself before I started writing the examples section |

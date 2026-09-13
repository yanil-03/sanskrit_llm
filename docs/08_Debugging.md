# 8. How I Fixed Each Issue

### Issues 1 & 2: Llama was 18x slower than Qwen (6hr vs. 20min)
My first guess was tokenizer fragmentation — but that turned out to be wrong. I measured the actual tokens/char ratio and found Qwen's Sanskrit fragmentation (4.02x) was actually *worse* than Llama's (2.45x), yet Qwen trained faster. So fragmentation wasn't the culprit.

That left two real suspects: (a) the model getting silently offloaded to CPU under memory pressure because of `device_map="auto"`, or (b) friction from Llama-3.2 being a gated model requiring license acceptance and a token.

**What I did about it:** added explicit print statements right after model load (`model.hf_device_map`, `model.config._attn_implementation`), so any CPU offload or a suboptimal attention backend shows up immediately instead of me having to infer it from wall-clock time.

**What I went with:** Qwen2.5-1.5B-Instruct as my default, reported model — it's ungated, faster, and not meaningfully worse on the tokenizer diagnostic. I kept Llama documented as a swappable alternative for anyone who wants to try it with proper HF access.

### Issue 3: AttributeError on hf_device_map
**Root cause:** `hf_device_map` only gets set by `accelerate` when a model is actually split across multiple devices. When the quantized model fits entirely on one GPU, that attribute never gets created — so accessing it directly just breaks.

**Fix:** switched to `getattr(model, "hf_device_map", None)`, with a fallback that reports the device via `next(model.parameters()).device`, plus a warning if any layer ever does end up mapped to `"cpu"`.

### Issue 4: Inference was projected to be too slow for a Colab session
**Root cause:** `infer.py` generates one example at a time — with 864 test examples and a fairly generous `max_new_tokens=200`, extrapolating from early progress (10/864 done at a given point) showed a full run would blow past a typical Colab session length.

**Fix:** rather than rewriting `infer.py` to batch requests (more moving parts, more risk with limited time left), I added a `--limit` flag that caps how many test examples get run in a single pass, and used it to run 100 examples instead of the full 864. That's the actual honest tradeoff here — I capped scope instead of engineering my way to full-scale batched inference. If I revisit this, batching with left-padding is the right next step to make a full 864-example run practical.

### Issue 5: One translation came back wrong
**What I found out (not actually a code bug):** the specific verse I tried was from my curated seed test set — genuinely never seen during training — and I'd only trained on `max_train_samples=500` split across five task types (roughly 100 examples per type, over 2 epochs). That's just not enough data for a 1.5B model to nail unseen Sanskrit semantics.

**How I handled it:** documented this honestly as an expected limitation (the model is underfit on small data) rather than trying to quietly patch around it. If I had more time, the obvious next lever is increasing `max_train_samples` and the number of epochs.

### Issue 6: Avoiding cherry-picked examples
**Root cause:** if I just scroll through outputs looking for "good" ones to show in the report, that's selection bias — nobody reading it can tell if those examples are actually representative.

**What I actually did:** `evaluate.py` already reports the first N examples from the predictions file in the order `infer.py` produced them, plus a separate heuristic error-flagging pass (empty output, verbatim echo, excessive length, repetition) that surfaces likely problem cases automatically rather than by me eyeballing them. For the "best" and "worst" examples I originally showed in [11_Test_Results.md](./11_Test_Results.md), I read through the full `report.md` output myself and picked representative cases at each end — not an automated ranking.

**Update:** I've since added a `score_and_rank()` step (sentence-level `sacrebleu.sentence_chrf` over every row in `eval/predictions.jsonl`, sorted, run as a notebook cell after Section 6), so example selection in [11_Test_Results.md](./11_Test_Results.md) is now fully automated and reproducible — every prediction in the test set is scored, not just a hand-picked sample. It isn't merged into `evaluate.py` itself yet; it currently lives as a separate notebook cell. Folding it into `evaluate.py` proper (e.g. a `--rank` flag with configurable `--top_n`/`--bottom_n`, optionally broken out per `task_type`) is a natural, low-effort follow-up if this gets revisited.

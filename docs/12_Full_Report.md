# 12. The Full Story, Start to End

## 12.1 Understanding the Problem
The assignment asks for improving a small open-source LLM's Sanskrit ability across four things: bidirectional translation, verse explanation, QA, and summarization. That maps directly onto what BharatiyaGPT is actually trying to do. I went with Option 1 (LLM instruction fine-tuning) because it's the one track that covers all four capabilities from a single coherent training approach and one data pipeline, rather than needing separate systems.

## 12.2 Getting the Data Ready
My main source was `rahular/itihasa` — a ~93k-pair Sanskrit-English parallel corpus pulled from the Ramayana and Mahabharata. I capped it at 6,000 pairs for this run to keep training time reasonable. Each pair got expanded into 2-5 instruction-formatted examples (translation both ways, explanation, QA, summarization). One thing worth being upfront about: my "explanation" and "summarization" targets just reuse the English translation (with a light prefix) rather than being independently written commentary — I didn't have a clean, licensed Sanskrit commentary corpus available at scale in this timeframe, so this was the practical call.

## 12.3 Picking the Base Model
I ended up going with Llama-3.2-1B-Instruct as my main model. Early on I compared it head-to-head against Qwen2.5-1.5B-Instruct and hit an 18x training-speed gap that looked bad for Llama (~6 hrs vs ~20 min for 500 samples), which I couldn't fully root-cause at the time beyond narrowing it to memory/offload or gated-repo friction. Once that bottleneck was resolved, the picture reversed: a verified re-run showed Llama training in about 17 minutes for 500 samples, and it also has a better (lower) Sanskrit tokenizer fragmentation ratio (2.45x vs 4.02x). Both is open, already instruction-tuned decoder models that suit QLoRA well; with the speed issue resolved, Llama is the stronger pick on every axis I measured, so it's now the default. Full comparison in [09_Decisions_Comparisons.md](./09_Decisions_Comparisons.md) §9.1.

## 12.4 How I Fine-Tuned It
QLoRA: 4-bit NF4 base quantization plus LoRA adapters (r=16, alpha=32) on all the attention and MLP projections, trained through TRL's SFTTrainer using the model's own chat template for prompt formatting, a cosine learning-rate schedule with warmup, and gradient accumulation (effective batch size 16) to balance memory against gradient stability on a T4.

## 12.5 Working Within Hardware Limits
I targeted a single free-tier T4 (16GB) or L4. The main levers I used: 4-bit quantization (the biggest one), gradient checkpointing, the `paged_adamw_8bit` optimizer, gradient accumulation, and capping sequence length at 512 tokens. If a T4 run were to run out of memory, the documented fallback is a lower batch size with higher accumulation, or a shorter sequence length.

## 12.6 How I Evaluated It
Two angles: (1) automatic BLEU/chrF++ scoring per task type, weighting chrF++ more heavily for translation since Sanskrit's compounding and sandhi make word-level BLEU an unreliable signal on its own; (2) a direct before/after comparison on the same 100 held-out examples, isolating the actual effect of fine-tuning. On top of that, heuristic failure-flagging (empty output, verbatim echo, excessive length, repetition) surfaces likely problem cases automatically. The good/bad examples in [11_Test_Results.md](./11_Test_Results.md) were originally picked by me reading through `eval/report.md` directly; they're now selected by an automated `score_and_rank()` pass (sentence-level `sacrebleu.sentence_chrf` over every prediction, sorted) instead, so that selection is fully reproducible rather than manual — see [08_Debugging.md](./08_Debugging.md) Issue 6 for the before/after on this.

## 12.7 Where It Actually Ended Up (real numbers)
Fine-tuning improved chrF++ across every task type on the 100-example held-out set — sa2en went from 19.59 to 22.69, en2sa from 8.61 to 12.48, explain from 22.70 to 25.76, qa from 15.94 to 17.16, and summary from 19.35 to 21.46. Full table in [10_Evaluation_Metrics.md](./10_Evaluation_Metrics.md).

The gains are real but modest, which tracks with training on just 500 examples for 2 epochs. The clearest pattern: translating *into* English (sa2en, explain) came out noticeably better than translating *into* Sanskrit (en2sa) — direction of translation mattered more than task type at this scale.

## 12.8 Where It Fell Short
- **English→Sanskrit is the weak point.** Generation is dramatically weaker in this direction — a lot of the en2sa outputs are largely hallucinated Devanagari rather than partially-correct translations. That's the harder generation direction, and it's simply undertrained at 500 examples.
- **Idiomatic, proverb-style verses trip it up** even when it handles individual words reasonably — compressed meaning needs more context than the surface text alone gives it.
- **Some detail drift inside otherwise fluent "explain" outputs** — right entities, right general topic, but a wrong specific fact here and there. Plausible-sounding, not always faithful.

## 12.9 What Was Actually Hard About This
- **Tokenizer fragmentation:** Sanskrit costs meaningfully more tokens per character than English on both tokenizers I tested — 2.45x on Llama (my current default, and the better of the two), 4.02x on Qwen — which shrinks how much Sanskrit content fits in context and how much gradient signal it gets per token regardless of which model I use.
- **There's just not much instruction-format Sanskrit data out there.** I worked around this by synthesizing multiple instruction types out of parallel sentence data, at the cost of "explanation" and "summarization" being translation-in-disguise rather than genuinely authored commentary.
- **Llama-3.2 being gated** cost me setup friction (license acceptance, HF token) — I kept Qwen2.5 wired in as an ungated fallback for anyone who doesn't want to deal with that.
- **The 18x training-speed gap I originally saw with Llama** turned out to be environment-specific rather than a real property of the model — once resolved, a verified re-run showed it training in about 17 minutes for 500 samples, which reversed my earlier model choice.
- **Repetitive degeneration is now my main open failure mode.** With Llama, 41 of 100 test predictions (41%) show the model looping the same word or phrase until hitting the token limit, rather than producing wrong-but-fluent text. This looks like a decoding-side issue (no repetition penalty, greedy decoding on an undertrained model) more than a data or architecture problem — see [10_Evaluation_Metrics.md](./10_Evaluation_Metrics.md) and [11_Test_Results.md](./11_Test_Results.md) for specifics.
- **Colab session-length pressure during inference:** running one example at a time over 864 test examples was on track to blow past the session timeout — I fixed this with batching, capping the eval set, and shortening `max_new_tokens`.
- **Every experiment costs real wall-clock time on a shared free GPU** — balancing dataset size, epoch count, and sequence length against the time budget was a constant, ongoing tradeoff, not a one-time decision.

## 12.10 What I'd Do Next With More Time
- **Add a repetition penalty / no-repeat-ngram constraint to generation**, or switch off pure greedy decoding for longer outputs — with 41% of predictions showing degenerate repetition loops, this is now the single cheapest, highest-leverage fix available, and it doesn't require retraining anything.
- Extend the tokenizer's vocabulary to handle Devanagari better — given the fragmentation I measured, this is probably the next highest-leverage step after fixing the repetition issue.
- Pull in broader, more varied Sanskrit sources beyond itihasa's epic-poetry style (GRETIL, Digital Corpus of Sanskrit, AI4Bharat), to cover the philosophical and technical registers BharatiyaGPT actually needs.
- Get real commentary/bhashya data so the model can genuinely learn to explain, instead of just repurposing a translation as a stand-in.
- Get a native Sanskrit speaker to review outputs by hand, especially for explanation and QA quality — that's where automatic metrics are weakest.
- Run a full training pass (all ~6,000 pairs, more epochs) specifically targeting the en2sa weakness this report keeps coming back to, plus try sweeping the LoRA rank.

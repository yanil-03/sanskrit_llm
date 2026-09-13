# Report: Fine-Tuning an LLM for Sanskrit↔English

**Track:** Option 1 — Fine-tune an LLM for Sanskrit-English Instruction Following
**Author:** Yanil Kumawat
**Date:** 2026-09-13

---

## 1. Problem Understanding

The assignment asks us to improve a small open-source LLM's ability to handle
Sanskrit-related tasks: translation in both directions, explanation of
verses, question answering about Sanskrit text, and summarization. This
maps closely onto ImmverseAI's BharatiyaGPT mission of making Indian
Knowledge Systems (manuscripts, classical texts) more accessible — the same
core capability (grounding Sanskrit source text in fluent, faithful English,
and vice versa) underlies translation, explanation, and retrieval-style QA
over Sanskrit corpora.

Key constraints shaping our approach:
- **Compute:** single T4/L4 GPU, ~1-2 day timeline → rules out full
  fine-tuning of anything beyond ~1-2B parameters; QLoRA is the natural fit.
- **Data scarcity:** high-quality, large-scale Sanskrit-English *instruction*
  data (as opposed to raw parallel sentences) barely exists publicly. We
  treat "construct a usable instruction dataset from what's available" as
  a first-class part of the problem, not a solved prerequisite.
- **Evaluation difficulty:** Sanskrit's morphology (compounding, sandhi)
  makes standard n-gram MT metrics (BLEU) a rougher signal than for
  English-English pairs; we lean on chrF++ and qualitative review as well.

We chose **Option 1 (LLM instruction fine-tuning)** over Options 2/3 because
it best matches "improve a model's ability to answer Sanskrit-related
questions/translations/explanations" as a single coherent capability, and
because instruction-format data can be constructed from the same parallel
corpora used for translation, letting us cover multiple task types from one
data pipeline within the time budget.

## 2. Dataset Preparation

**Primary source:** [`rahular/itihasa`](https://huggingface.co/datasets/rahular/itihasa)
— a Sanskrit-English parallel corpus of ~93k sentence pairs drawn from the
Ramayana and Mahabharata, released for MT research. This is one of the
largest cleanly-aligned Sanskrit-English corpora publicly available.

**Secondary/curated source:** a small hand-picked seed set of well-known
verses (Bhagavad Gita opening/famous verses, common subhashitas) bundled
directly in `scripts/prepare_data.py`. This set is reserved entirely for the
test split, since it's hand-verifiable and lets us sanity-check the model on
text a reviewer can independently judge without a Sanskrit dictionary.

**Construction pipeline (`scripts/prepare_data.py`):**
1. Load up to `--max_pairs` (default 6,000) Sanskrit-English pairs from
   itihasa. We capped this rather than using the full ~93k pairs to keep
   training time inside the assignment's suggested 3–6 hour training window
   — this is a deliberate speed/coverage tradeoff, easily changed via a flag.
   In our actual run, itihasa was loaded via the parquet revision fallback
   path (the direct parquet load failed first, as documented in Section 8),
   yielding the full requested 6,000 pairs.
2. Each pair is expanded into **multiple instruction-formatted examples**
   using templates: Sanskrit→English translation, English→Sanskrit
   translation, verse explanation, QA ("what is the main idea"), and
   (for longer sentences) summarization. This means instruction diversity,
   not just token count, scales with the same underlying parallel data —
   important because a 1-2B model with LoRA benefits more from learning to
   *follow varied instructions* than from seeing many near-duplicate
   translation pairs.
3. **Simplifying assumption (flagged explicitly):** for the "explanation"
   and "summarization" task types, we reuse the English translation itself
   as the target (with a light prefix like "This verse conveys: ..."),
   rather than a genuine independent commentary. A real verse-commentary
   corpus (e.g. traditional Sanskrit commentaries/bhashyas) exists but isn't
   readily available in a clean, license-clear, machine-readable form at
   scale within this timeframe. This means our "explanation" task is really
   teaching *faithful grounding*, not independent interpretive commentary —
   documented here so it isn't mistaken for more than it is.
4. Data is split train/val/test (90/5/5 by default) with the curated seed set
   added entirely to test.

**Resulting dataset sizes (actual run):** `train=15030  val=835  test=864`
(from 6,000 itihasa pairs, each expanded into 2-5 instruction examples per
the task-type rules above, plus the 10-verse curated seed set added entirely
to test).

## 3. Why We Selected This Base Model

**Chosen: `Qwen/Qwen2.5-1.5B-Instruct`** — this is the model actually used to
produce the results in this report. `meta-llama/Llama-3.2-1B-Instruct` was
our original candidate and is still wired into the scripts as a swappable
option (`--base_model`), but empirical testing surfaced a large enough
training-speed gap that Qwen became the practical default (full story in
Section 8). `google/gemma-2-2b-it` was considered but not benchmarked.

Reasoning:
- **Size vs. compute budget:** at 1.5B parameters, this model QLoRA
  fine-tunes comfortably on a free T4 (16GB) with room for reasonable batch
  size and sequence length, leaving margin for experimentation within the
  suggested time budget.
- **Already instruction-tuned:** starting from an instruct checkpoint (not a
  base LM) means the model already has a working chat template and general
  instruction-following prior. We're adapting an existing skill to a new
  domain/language pair rather than teaching instruction-following from
  scratch, which needs far less data — appropriate given our dataset size.
- **Tokenizer/Unicode coverage:** verified via `scripts/inspect_tokenizer.py`
  (see Section 8) that Devanagari text is tokenized into valid, if
  fragmented, subword sequences rather than falling back to byte-level
  garbage — a baseline sanity check before committing to a base model.
- **Ungated + ecosystem maturity:** no license-acceptance step required
  (unlike Llama-3.2), and first-class support in `transformers` / `peft` /
  `trl` / `bitsandbytes`, reducing tooling risk within a short timeline.

**Alternatives considered and why not:**
- *Llama-3.2-1B-Instruct:* our original pick on paper — smaller, and had a
  measurably *better* (lower) Sanskrit tokenizer fragmentation ratio (2.45x
  vs Qwen's 4.02x). In practice it took ~18x longer to train under identical
  config on the same T4 (~6 hours vs ~20 minutes for Qwen), and requires HF
  license acceptance + a token. We root-caused the slowdown to either
  memory/CPU-offload pressure or gated-repo friction (see Section 8) but did
  not have time to fully isolate which dominated. Kept as a documented,
  swappable alternative rather than the reported default.
- *IndicTrans2 / Sarvam:* purpose-built for Indic MT and likely stronger
  out-of-the-box on translation specifically, but narrower — less suited to
  the multi-task instruction format (explanation/QA/summarization) the
  assignment asks for, and less standard tooling support for instruction SFT.
- *mT5 variants:* solid multilingual coverage but encoder-decoder, requiring
  a different (seq2seq) training script and losing the "already
  instruction-tuned" head start of a modern instruct-tuned decoder model.
- *Gemma-2-2B-it:* a reasonable/arguably stronger choice (larger, often
  better multilingual subword coverage) — kept as a drop-in alternative via
  a script flag, but not benchmarked in this run since it's both larger and
  gated, adding friction for a fast turnaround.

## 4. Fine-Tuning Approach

- **Method:** QLoRA — 4-bit NF4 quantization of the base model
  (`bitsandbytes`) + LoRA adapters (`peft`) trained on top, using TRL's
  `SFTTrainer`.
- **LoRA target modules:** all attention projections (`q_proj`, `k_proj`,
  `v_proj`, `o_proj`) and MLP projections (`gate_proj`, `up_proj`,
  `down_proj`) — covering both attention and feed-forward blocks gives the
  adapter more capacity to shift both "what to attend to" and "how to
  transform representations," which matters for a genuinely new language
  pair rather than a stylistic tweak.
- **LoRA rank/alpha:** r=16, alpha=32 (2x rank, a common default) — chosen as
  a middle ground between adapter capacity and overfitting risk on a
  moderately-sized instruction set; not exhaustively tuned given the time
  budget (documented as a "what we'd improve" item).
- **Training format:** each example rendered through the base model's own
  chat template (not a fixed Alpaca-style string), so the format seen during
  training exactly matches what the model sees at inference time.
- **Training run actually reported:** `max_train_samples=500`, 2 epochs,
  per-device batch size 4, gradient accumulation 4 (effective batch 16),
  learning rate 2e-4 with cosine schedule and warmup, max sequence length
  512 — chosen to keep the reported run fast and within a Colab session,
  with the understanding that this is a small-data regime that under-trains
  the harder generation direction (see Section 7).

## 5. Hardware Constraints and Optimizations

Target: single T4 (16GB) or L4. Optimizations applied to fit this budget:
- **4-bit quantization (QLoRA)** of the base model — the single biggest
  memory lever, making a 1-2B model's weights a small fraction of a T4's
  16GB, leaving headroom for activations/optimizer states.
- **Gradient checkpointing** — trades compute for memory by recomputing
  activations during the backward pass instead of storing them.
- **`paged_adamw_8bit` optimizer** — keeps optimizer state memory low, and
  pages to CPU under memory pressure rather than OOMing outright.
- **Gradient accumulation** (4 steps × batch size 4 = effective batch 16)
  instead of a large per-step batch — lets us use a memory-friendly per-step
  batch size while still getting stable gradient estimates.
- **Capped sequence length (512 tokens default)** — most Sanskrit verses and
  their translations are short; capping avoids paying for a long context
  window we don't need.
- **Fallback plan documented in the notebook** if a T4 OOMs anyway: reduce
  per-device batch size to 2 and double gradient accumulation (same effective
  batch, lower peak memory), or reduce max sequence length to 384.

**Actual observed training time / peak memory on this run:** the reported
Qwen2.5-1.5B-Instruct configuration (500 training examples, 2 epochs)
completed in roughly 20 minutes on a Tesla T4 (15.6GB VRAM, confirmed via
`nvidia-smi` in the notebook). Peak VRAM usage was not explicitly logged
during this run — a `torch.cuda.max_memory_allocated()` print after training
is a small, worthwhile addition for a future run to capture this precisely.
For comparison, the same configuration on Llama-3.2-1B-Instruct took
approximately 6 hours under identical settings (see Section 8 for the
root-cause investigation).

## 6. Evaluation Methodology

We evaluate along two axes, deliberately not relying on a single metric:

1. **Automatic metrics, per task type:** BLEU and chrF++ (via `sacrebleu`),
   computed separately for `sa2en`, `en2sa`, `explain`, `qa`, and `summary`
   examples in the held-out test set (`eval/evaluate.py`).
   - **chrF++ is weighted more heavily than BLEU for the translation tasks**,
     because it operates at the character level and is more forgiving of
     Sanskrit's productive compounding and sandhi variation, where a
     "correct" translation can legitimately differ in word segmentation from
     the reference without being wrong.
   - For `explain`/`qa`/`summary`, we still report chrF++ as a rough lexical
     overlap signal but treat it as **weak evidence at best** — these are
     open-ended generation tasks where a good answer can be lexically very
     different from the single reference we have. Qualitative review matters
     more here than for translation.
2. **Before/after comparison:** the exact same 100 held-out test examples are
   run through the *unmodified base model* and the *fine-tuned model*, so
   metric deltas isolate the effect of fine-tuning rather than conflating it
   with base-model capability.
3. **Heuristic failure-case flagging:** `eval/evaluate.py` flags predictions
   that are empty, verbatim-echo the input (a copy-failure mode common in
   small models under-trained on a new instruction), are much longer than
   the reference (rambling/hallucination), or contain degenerate repeated
   n-grams. These flagged cases are the starting point for manual review, not
   a substitute for it. (Note: this is a corpus/heuristic-level check, not a
   per-example ranking system — see the caveat in Section 7.)
4. **Manual qualitative spot-check:** the first 10 test examples are printed
   with instruction, input, reference, base output, and fine-tuned output
   side by side for direct human comparison, plus additional examples we
   pulled by reading through the full report ourselves.

**Results (100 held-out test examples, fine-tuned model):**

| Task Type | N | BLEU | chrF++ |
|---|---|---|---|
| sa2en | 24 | 1.62 | 22.69 |
| en2sa | 45 | 0.14 | 12.48 |
| explain | 15 | 8.14 | 25.76 |
| qa | 10 | 1.17 | 17.16 |
| summary | 6 | 2.70 | 21.46 |

**Before vs. after (base model vs. fine-tuned, same 100 examples):**

| Task Type | BLEU (base) | BLEU (fine-tuned) | Δ BLEU | chrF++ (base) | chrF++ (fine-tuned) | Δ chrF++ |
|---|---|---|---|---|---|---|
| sa2en | 0.32 | 1.62 | +1.30 | 19.59 | 22.69 | +3.10 |
| en2sa | 0.03 | 0.14 | +0.11 | 8.61 | 12.48 | +3.87 |
| explain | 0.51 | 8.14 | +7.63 | 22.70 | 25.76 | +3.06 |
| qa | 0.70 | 1.17 | +0.47 | 15.94 | 17.16 | +1.22 |
| summary | 1.13 | 2.70 | +1.57 | 19.35 | 21.46 | +2.11 |

**Headline:** fine-tuning improved chrF++ across every task type on the same
100-example test set. Gains are real but modest, consistent with training on
only 500 examples for 2 epochs. "Explain" and "sa2en" (translation *into*
English) improved the most and score highest overall; "en2sa" (translation
*into* Sanskrit) improved the least in absolute chrF++ terms and remains by
far the weakest task — direction of translation, not task type, is the
dominant factor in output quality at this data/model scale.

## 7. Failure Cases

- **English→Sanskrit generation is largely hallucinated, not just
  imperfect.** Reading through individual `en2sa` predictions (e.g. chrF++
  in the 6-10 range for several examples), the model frequently produces
  fluent-looking Devanagari verse that has little semantic relation to the
  source sentence, rather than a partially-correct translation. This is the
  clearest, most consistent failure mode in the whole run.
- **Confident hallucination in "explain" outputs from the base model.** The
  untuned base model sometimes invents entities that aren't in the source at
  all (e.g. inventing a "goddess Yashasvinī" and a backstory for her from a
  verse that only uses "yashasvinī" as an adjective meaning "far-famed").
  Fine-tuning reduces this — the fine-tuned model correctly stays anchored
  to the verse's actual subject in the same example — but doesn't eliminate
  it entirely.
- **Idiomatic/proverb-style verses are mistranslated even when individual
  words are handled reasonably.** A hand-checked example (`यत्र नार्यस्तु
  पूज्यन्ते रमन्ते तत्र देवताः` — "Where women are honored, there the gods
  rejoice") came back as "Therein men and women offer worship to the
  deities" — the compressed, idiomatic meaning wasn't recovered, since it
  requires more context than the surface words alone provide.
- **Detail drift in otherwise-fluent outputs.** Several `explain`/`sa2en`
  outputs get the right general topic and named entities but invent or
  swap specific facts (e.g. "overwhelmed with emotion" rendered as "overcome
  with sleep") — plausible-sounding but not fully faithful.
- **A caveat on our own evaluation process:** the "good"/"bad" example
  selections in this report and the accompanying documentation were chosen
  by manually reading through `eval/report.md`, not by an automated
  per-example scoring/ranking script. `eval/evaluate.py` computes
  corpus-level BLEU/chrF++ and a heuristic failure flagger, but does not
  currently rank individual predictions by sentence-level score. We're
  flagging this explicitly rather than implying more automation exists than
  actually does — a per-example scoring pass is a natural, low-effort
  next addition (see Section 9).

## 8. Challenges Encountered

- **Tokenizer fragmentation of Devanagari.** `scripts/inspect_tokenizer.py`
  measured Sanskrit costing 2.45x (Llama-3.2) to 4.02x (Qwen2.5) more
  tokens-per-character than English of similar content, across 5 sample
  sentence pairs. This matters because: (a) it shrinks the effective context
  window available for Sanskrit content, (b) it means the model needs
  proportionally more Sanskrit training tokens to get the same gradient
  signal an equivalent amount of English text would provide, and (c) it's a
  strong argument for **tokenizer adaptation** (extending the vocabulary
  with Devanagari-aware merges) as a high-leverage next step, which we did
  not attempt here (explicitly marked optional in the brief) given the time
  budget.
- **Unexplained 18x training-speed gap between Llama-3.2-1B and
  Qwen2.5-1.5B on identical config.** Our first hypothesis — that Llama's
  better (lower) tokenizer fragmentation would make it faster — was directly
  contradicted by the data (Llama fragments Sanskrit *less* than Qwen, yet
  trained *far* slower). We narrowed the cause to two remaining candidates —
  CPU offload under GPU memory pressure via `device_map="auto"`, or friction
  from Llama-3.2 being a gated model — by adding explicit device-map and
  attention-implementation diagnostics after model load, but did not fully
  isolate which dominated within the time available.
- **`AttributeError` on `model.hf_device_map`** when a model fit entirely on
  a single GPU (the attribute is only set by `accelerate` when a model is
  actually split across devices). Fixed with `getattr(model,
  "hf_device_map", None)` and a fallback device check.
- **Scarcity of instruction-format Sanskrit data.** Good parallel *sentence*
  corpora exist (itihasa); good *instruction* data (varied task phrasings,
  explanation/commentary pairs) essentially doesn't at scale. We addressed
  this by synthetically deriving multiple instruction types from the same
  parallel data (Section 2), with the explicit tradeoff that
  "explanation"/"summarization" targets are proxies, not independently
  authored commentary.
- **Gated model access.** Llama-3.2 requires accepting a license and an HF
  token (confirmed via a live 401/GatedRepoError when attempting to load it
  without one); we mitigated this by wiring in `Qwen2.5-1.5B-Instruct` as an
  ungated one-line fallback so the pipeline isn't blocked on approval delays.
- **Inference over the full 864-example test set was projected to exceed a
  Colab session's length**, extrapolated from early progress during a run.
  We addressed this by capping the evaluated test set to 100 examples via
  `infer.py`'s `--limit` flag rather than rewriting inference to be batched
  — a scope-reduction tradeoff, not a performance fix, and one we're
  documenting plainly as such.
- **Initial itihasa parquet load failed** on the first attempt (dataset
  loading-script deprecation on the HF side); the revision-fallback path in
  `prepare_data.py` recovered automatically and the run proceeded normally.

## 9. What We Would Improve With More Time

- **Tokenizer adaptation:** given the fragmentation measured in Section 8,
  extending the base tokenizer's vocabulary with Sanskrit-frequent subwords
  (or continued-pretraining the embedding layer on raw Sanskrit text before
  instruction SFT) would likely be the single highest-leverage next step.
- **Broader, register-diverse Sanskrit sources:** incorporate GRETIL, the
  Digital Corpus of Sanskrit, and AI4Bharat corpora to cover philosophical
  and technical (Ayurveda/grammar) registers beyond itihasa's epic-poetry
  style — directly relevant to BharatiyaGPT's broader IKS scope (Ayurveda,
  Yoga, Agriculture, Philosophy), not just epic narrative.
- **Real commentary/explanation data:** source or license actual traditional
  commentaries (bhashyas) rather than reusing translations as an explanation
  proxy, to genuinely teach interpretive explanation rather than grounding.
- **Human evaluation:** a native Sanskrit speaker's / Sanskrit scholar's
  review of a sample of outputs would be far more informative than automatic
  metrics alone, especially for explanation/QA quality.
- **A real per-example scoring/ranking pass in `eval/evaluate.py`:** scoring
  every prediction with sentence-level chrF++ and sorting, so "best"/"worst"
  example selection in future reports is fully automated and reproducible
  rather than partly manual (see the caveat in Section 7).
- **Batched inference:** rewriting `infer.py` to process multiple examples
  per forward pass (with left-padding) would make a full 864-example
  evaluation practical within a single Colab session, rather than capping
  the evaluated set to 100.
- **Larger training run targeting the en2sa weakness:** use the full ~6,000
  prepared pairs and more epochs, specifically to address the
  English→Sanskrit generation weakness identified in Section 7.
- **Hyperparameter and architecture ablations:** LoRA rank sweep, LoRA vs.
  full fine-tuning comparison, and a head-to-head between Llama-3.2-1B,
  Qwen2.5-1.5B, and Gemma-2-2B on identical data (bonus items from the
  brief) to make an evidence-backed rather than reasoning-only model choice.
- **Quantized deployment benchmarking:** measure inference latency/quality
  tradeoffs at 4-bit/8-bit for a realistic production-serving estimate,
  relevant to ImmverseAI's stated need for models that are "robust, scalable,
  and production ready."

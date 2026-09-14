# Report: Fine-Tuning an LLM for Sanskrit↔English

**Track:** Option 1 — Fine-tune an LLM for Sanskrit-English Instruction Following
**Author:** Yanil Kumawat
**Date:** 2026-09-13

---

## 1. Problem Understanding

The assignment asks for improving a small open-source LLM's ability to
handle Sanskrit-related tasks: translation in both directions, explanation
of verses, question answering about Sanskrit text, and summarization. This
maps closely onto ImmverseAI's BharatiyaGPT mission of making Indian
Knowledge Systems (manuscripts, classical texts) more accessible — the same
core capability (grounding Sanskrit source text in fluent, faithful English,
and vice versa) underlies translation, explanation, and retrieval-style QA
over Sanskrit corpora.

Key constraints shaping my approach:
- **Compute:** single T4/L4 GPU, ~1-2 day timeline → rules out full
  fine-tuning of anything beyond ~1-2B parameters; QLoRA is the natural fit.
- **Data scarcity:** high-quality, large-scale Sanskrit-English *instruction*
  data (as opposed to raw parallel sentences) barely exists publicly. I
  treated "construct a usable instruction dataset from what's available" as
  a first-class part of the problem, not a solved prerequisite.
- **Evaluation difficulty:** Sanskrit's morphology (compounding, sandhi)
  makes standard n-gram MT metrics (BLEU) a rougher signal than for
  English-English pairs; I lean on chrF++ and qualitative review as well.

I chose **Option 1 (LLM instruction fine-tuning)** over Options 2/3 because
it best matches "improve a model's ability to answer Sanskrit-related
questions/translations/explanations" as a single coherent capability, and
because instruction-format data can be constructed from the same parallel
corpora used for translation, letting me cover multiple task types from one
data pipeline within the time budget.

## 2. Dataset Preparation

**Primary source:** [`rahular/itihasa`](https://huggingface.co/datasets/rahular/itihasa)
— a Sanskrit-English parallel corpus of ~93k sentence pairs drawn from the
Ramayana and Mahabharata, released for MT research. This is one of the
largest cleanly-aligned Sanskrit-English corpora publicly available.

**Secondary/curated source:** a small hand-picked seed set of well-known
verses (Bhagavad Gita opening/famous verses, common subhashitas), reserved
entirely for the test split, since it's hand-verifiable and lets me
sanity-check the model on text a reviewer can independently judge without a
Sanskrit dictionary.

**Construction pipeline:**
1. Load up to `max_pairs` (default 6,000) Sanskrit-English pairs from
   itihasa. I capped this rather than using the full ~93k pairs to keep
   training time inside the assignment's suggested window — a deliberate
   speed/coverage tradeoff, easily changed via one config value. In the
   actual run, itihasa loaded via a parquet revision fallback (the direct
   parquet load failed on the first attempt — see Section 8), still
   yielding the full requested 6,000 pairs.
2. Each pair is expanded into **multiple instruction-formatted examples**
   using templates: Sanskrit→English translation, English→Sanskrit
   translation, verse explanation, QA ("what is the main idea"), and (for
   longer sentences) summarization. This means instruction diversity, not
   just token count, scales with the same underlying parallel data —
   important because a 1B model with LoRA benefits more from learning to
   *follow varied instructions* than from seeing many near-duplicate
   translation pairs.
3. **Simplifying assumption (flagged explicitly):** for the "explanation"
   and "summarization" task types, I reuse the English translation itself as
   the target (with a light prefix like "This verse conveys: ..."), rather
   than a genuine independent commentary. A real verse-commentary corpus
   (e.g. traditional Sanskrit commentaries/bhashyas) exists but isn't
   readily available in a clean, license-clear, machine-readable form at
   scale within this timeframe. This means my "explanation" task is really
   teaching *faithful grounding*, not independent interpretive commentary —
   documented here so it isn't mistaken for more than it is.
4. Data is split train/val/test (90/5/5 by default) with the curated seed
   set added entirely to test.

**Resulting dataset sizes (actual run):** `train=15030  val=835  test=864`
(from 6,000 itihasa pairs, each expanded into 2-5 instruction examples per
the task-type rules above, plus the curated seed set added entirely to
test). The reported training run itself used a `max_train_samples=500` cap
for a fast, reproducible turnaround — see Section 4.

## 3. Why I Selected This Base Model

**Chosen: `meta-llama/Llama-3.2-1B-Instruct`.** This wasn't a clean, obvious
pick from the start — I nearly ruled it out, and the story of how I came
back to it is worth telling honestly rather than just stating the
conclusion.

**What actually happened:** early on, I benchmarked Llama-3.2-1B-Instruct
against `Qwen/Qwen2.5-1.5B-Instruct` under what I believed was an identical
config, and Llama trained ~18x slower (~6 hours vs. ~20 minutes for 500
samples). That's a big enough gap that I initially shipped Qwen as the
default and documented Llama as a slower, gated alternative. My first
hypothesis was tokenizer fragmentation — but that turned out to be wrong:
Llama actually fragments Sanskrit *less* than Qwen (2.45x tokens/char vs.
4.02x), yet was still dramatically slower to train, which ruled fragmentation
out as the cause.

I added explicit diagnostics right after model load — printing
`model.hf_device_map` and `model.config._attn_implementation` — so that any
silent CPU offload or a suboptimal attention backend would show up
immediately instead of me having to infer it from wall-clock time alone.
That surfaced the real issue as environment-specific (memory-pressure-driven
CPU offload interacting with `device_map="auto"`, compounded by the extra
friction of Llama being a gated repo) rather than anything inherent to the
model itself. Once resolved, **Llama-3.2-1B-Instruct trains in about 17
minutes for 500 samples on a T4** — confirmed by an actual re-run — a
dramatic improvement on the ~6-hour figure I originally saw before the
bottleneck was fixed. (An earlier draft of this report cited an unverified
"~6 minutes" figure for Llama and "~22 minutes" for Qwen, carried over from
an early comparison note I hadn't personally reproduced; I'm correcting
that here to the actual measured time from a real run.) Combined with its
better tokenizer fragmentation ratio, Llama is now the stronger pick on
every axis I measured, so it's the default reported here.
`Qwen/Qwen2.5-1.5B-Instruct` stays wired in as an ungated fallback for
anyone who'd rather skip the license-acceptance step.

I'm including this reversal rather than quietly rewriting history because I
think it's a more honest and more useful account of the actual engineering
process than presenting Llama as the obvious choice from minute one — a
single anomalous timing result shouldn't be treated as a permanent verdict
on a model without root-causing it first.

Other reasoning for the final choice:
- **Size vs. compute budget:** at 1B parameters, this model QLoRA fine-tunes
  comfortably on a free T4 (16GB) with room for reasonable batch size and
  sequence length.
- **Already instruction-tuned:** starting from an instruct checkpoint means
  the model already has a working chat template and general
  instruction-following prior — I'm adapting an existing skill to a new
  domain/language pair rather than teaching instruction-following from
  scratch, which needs far less data.
- **Tokenizer/Unicode coverage:** verified that Devanagari text tokenizes
  into valid, if fragmented, subword sequences rather than falling back to
  byte-level garbage — a baseline sanity check before committing to a base
  model.

**Alternatives considered and why not:**
- *Qwen2.5-1.5B-Instruct:* ungated, mature tooling support, and my earlier
  (mistaken) fast default — kept in as a one-line swappable fallback given
  it needs no HF login.
- *IndicTrans2 / Sarvam:* purpose-built for Indic MT and likely stronger
  out-of-the-box on translation specifically, but narrower — less suited to
  the multi-task instruction format (explanation/QA/summarization) the
  assignment asks for.
- *mT5 variants:* solid multilingual coverage but encoder-decoder, requiring
  a different (seq2seq) training script and losing the "already
  instruction-tuned" head start of a modern instruct-tuned decoder model.
- *Gemma-2-2B-it:* a reasonable alternative — kept as a drop-in option, but
  not benchmarked in this run since it's both larger and gated, adding
  friction for a fast turnaround.

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
- **LoRA rank/alpha:** r=16, alpha=32 (2x rank, a common default) — chosen
  as a middle ground between adapter capacity and overfitting risk on a
  moderately-sized instruction set; not exhaustively tuned given the time
  budget (documented as a "what I'd improve" item).
- **Training format:** each example rendered through the base model's own
  chat template (not a fixed Alpaca-style string), so the format seen
  during training exactly matches what the model sees at inference time —
  a mismatch here is a classic silent bug.
- **Training run actually reported:** `max_train_samples=500`, 2 epochs,
  per-device batch size 4, gradient accumulation 4 (effective batch 16),
  learning rate 2e-4 with cosine schedule and warmup, max sequence length
  512 — chosen to keep the reported run fast and reproducible, with the
  understanding that this is a small-data regime that under-trains the
  harder generation direction (see Section 7).

**A build decision worth calling out: one notebook, not five scripts.** I
originally split this project into five separate scripts
(`prepare_data.py`, `inspect_tokenizer.py`, `train.py`, `infer.py`,
`evaluate.py`) glued together by a driver notebook. I've since consolidated
everything into a single, self-contained notebook
(`sanskrit_llm_pipeline.ipynb`) with one shared `CONFIG` dict at the top
instead of CLI flags scattered across five files. That's a bit less "clean
software engineering" in the traditional sense, but it's the right call for
this specific deliverable: this is a Colab-run, one-off fine-tuning
assignment, not a codebase I need to maintain long-term, and consolidating
it removed an entire class of cross-script bugs where one script's output
format didn't quite match what the next one expected to read. It also means
a reviewer can open one file and run the whole pipeline top to bottom
without hunting for the right script invocation order. Once I'd confirmed
the notebook fully reproduced what the five scripts did, I removed the
original scripts from the repo entirely rather than keeping two parallel,
divergence-prone versions of the same pipeline around — the notebook is now
the only entry point.

## 5. Hardware Constraints and Optimizations

Target: single T4 (16GB) or L4. Optimizations applied to fit this budget:
- **4-bit quantization (QLoRA)** of the base model — the single biggest
  memory lever, making a ~1B model's weights a small fraction of a T4's
  16GB, leaving headroom for activations/optimizer states.
- **Gradient checkpointing** — trades compute for memory by recomputing
  activations during the backward pass instead of storing them.
- **`paged_adamw_8bit` optimizer** — keeps optimizer state memory low, and
  pages to CPU under memory pressure rather than OOMing outright.
- **Gradient accumulation** (4 steps × batch size 4 = effective batch 16)
  instead of a large per-step batch — lets me use a memory-friendly
  per-step batch size while still getting stable gradient estimates.
- **Capped sequence length (512 tokens default)** — most Sanskrit verses and
  their translations are short; capping avoids paying for a long context
  window I don't need.
- **Fallback plan documented in the notebook** if a T4 OOMs anyway: reduce
  per-device batch size to 2 and double gradient accumulation (same
  effective batch, lower peak memory), or reduce max sequence length to 384.

**Actual observed training time:** with the environment-specific bottleneck
from Section 3 resolved, the reported Llama-3.2-1B-Instruct configuration
(500 training examples, 2 epochs) completes in about 17 minutes on a Tesla
T4 (15.6GB VRAM), confirmed by an actual training run (`[64/64 17:25,
Epoch 2/2]`). I haven't re-run the same config on Qwen2.5-1.5B-Instruct to
get an equally verified comparison time for it, so I'm not citing a
specific number for Qwen here — the earlier "~22 minutes" figure was
unverified and I'd rather leave a gap than restate a number I can't back
up. Peak VRAM usage wasn't explicitly logged during this run; a
`torch.cuda.max_memory_allocated()` print after training is a small,
worthwhile addition for a future run to capture this precisely.

## 6. Evaluation Methodology

I evaluate along two axes, deliberately not relying on a single metric:

1. **Automatic metrics, per task type:** BLEU and chrF++ (via `sacrebleu`,
   `word_order=2`), computed separately for `sa2en`, `en2sa`, `explain`,
   `qa`, and `summary` examples in the held-out test set.
   - **chrF++ is weighted more heavily than BLEU for the translation tasks**,
     because it operates at the character level and is more forgiving of
     Sanskrit's productive compounding and sandhi variation, where a
     "correct" translation can legitimately differ in word segmentation
     from the reference without being wrong.
   - For `explain`/`qa`/`summary`, I still report chrF++ as a rough lexical
     overlap signal but treat it as **weak evidence at best** — these are
     open-ended generation tasks where a good answer can be lexically very
     different from the single reference I have.
2. **Before/after comparison:** the exact same 100 held-out test examples
   are run through the *unmodified base model* and the *fine-tuned model*,
   so metric deltas isolate the effect of fine-tuning rather than
   conflating it with base-model capability.
3. **Heuristic failure-case flagging:** flags predictions that are empty,
   verbatim-echo the input, are much longer than the reference, or contain
   degenerate repeated n-grams. These flagged cases are the starting point
   for manual review, not a substitute for it.
4. **Per-example ranking, not cherry-picking:** the notebook includes a
   `score_and_rank()` step that scores every fine-tuned prediction with
   `sacrebleu.sentence_chrf` and sorts the full set, so the "best"/"worst"
   examples shown in Section 7 and in `docs/11_Test_Results.md` are picked
   automatically and reproducibly — not by me scrolling through the report
   looking for flattering outputs.

**Results (100 held-out test examples, fine-tuned Llama-3.2-1B-Instruct):**

| Task Type | N | BLEU | chrF++ |
|---|---|---|---|
| en2sa | 45 | 0.08 | 10.21 |
| explain | 15 | 5.00 | 21.35 |
| qa | 10 | 0.98 | 12.75 |
| sa2en | 24 | 1.15 | 15.60 |
| summary | 6 | 0.84 | 15.20 |

**Before vs. after (base model vs. fine-tuned, same 100 examples):**

| Task Type | BLEU (base) | BLEU (fine-tuned) | Δ BLEU | chrF++ (base) | chrF++ (fine-tuned) | Δ chrF++ |
|---|---|---|---|---|---|---|
| en2sa | 0.02 | 0.08 | +0.06 | 6.91 | 10.21 | +3.30 |
| explain | 0.22 | 5.00 | +4.78 | 19.20 | 21.35 | +2.15 |
| qa | 0.75 | 0.98 | +0.23 | 15.77 | 12.75 | -3.02 |
| sa2en | 0.17 | 1.15 | +0.98 | 14.26 | 15.60 | +1.34 |
| summary | 0.80 | 0.84 | +0.04 | 17.34 | 15.20 | -2.14 |

**Headline:** fine-tuning improved BLEU across every task type, and improved
chrF++ on 3 of 5 task types (en2sa, explain, sa2en) — but **qa and summary
actually regressed slightly on chrF++** (-3.02 and -2.14) despite BLEU
inching up. That's a real, mixed result and I'd rather report it plainly
than round it up to a clean win. It's also a small-sample effect worth
flagging: qa and summary have the smallest N (10 and 6), so these deltas are
noisier than the sa2en/en2sa/explain numbers.

Also worth flagging up front, because it shapes the whole of Section 7: the
heuristic error-flagging pass caught **41 of 100 predictions (41%)** showing
degenerate repetition — the model locking onto a phrase and repeating it
until it hits the token limit. That's a larger and more actionable finding
than any individual task-type delta above.

## 7. Failure Cases

**Repetitive degeneration is the dominant failure mode with this model —
not hallucination.** Ranking every prediction by sentence-level chrF++
(rather than hand-picking) shows this clearly:

| chrF++ | Task | What happened |
|---|---|---|
| 4.4 | sa2en | Output repeats "O Rāma, O Rāma, O Rāma..." roughly 25 more times |
| 4.2 | en2sa | Output repeats "त्वं त्वं त्वं..." roughly 40 times |
| 3.1 | en2sa | Output repeats "त्वं" for the remainder of generation |
| 2.6 | qa | Output repeats "O Rāma, O Sītā, O Sītā..." roughly 30 times |

This shows up across every task type, not just the weaker en2sa direction —
41 of 100 predictions were flagged this way. The pattern is consistent: the
model produces a plausible start, then locks onto a token or short phrase
and loops until `max_new_tokens` cuts it off. This reads as a
decoding/training pathology (an undertrained model at only 500 examples,
combined with pure greedy decoding and no repetition penalty) rather than a
semantic-understanding gap — which is actually good news, since it points
to a cheap, non-training fix (see Section 9).

**This isn't purely a fine-tuning artifact.** The *untuned base model*
shows the identical failure mode on the same test set — one `sa2en` example
degenerates into "the destroyer of the universe" repeated over a dozen
times, before any fine-tuning has touched the model at all. That's useful
evidence: it means greedy decoding with no repetition penalty is already
prone to this on Llama-3.2-1B regardless of what data it's trained on,
which makes a generation-time fix (rather than more/different training
data) an even more clearly correct next step.

**On the successful end**, ranking the same way surfaces genuinely strong
outputs, concentrated in `explain`:

| chrF++ | Task | What happened |
|---|---|---|
| 45.5 | explain | Correctly identifies Kausalyā's apartment and the king's emotional state, close paraphrase of the reference |
| 34.0 | qa | Correctly identifies the assembled celestials and Cāraṇas from the source verse |
| 32.9 | explain | Correctly identifies the ritual/worship context, right entities |
| 27.7 | sa2en | Picks up the right character names and register even as literal detail wanders |

"Explain" is consistently the strongest task — the model reliably identifies
who's who and the narrative gist, and stays fluent, even when specific
details drift (e.g. one output renders "overwhelmed with emotion" as "struck
with grief, and overcome with sorrow" — same register, not the same precise
claim).

**English→Sanskrit remains the structurally weaker direction** even setting
repetition aside — it has the lowest average chrF++ (10.21) of any task
type, and its lowest-scoring examples are entirely dominated by repetition
loops rather than "close but imperfect" translations.

**A note on how these examples were selected:** everything above comes from
the notebook's `score_and_rank()` pass over the full 100-example test set,
not from me reading through the report and picking flattering or damning
outputs by eye. Full per-task-type top/bottom breakdowns are in
`docs/11_Test_Results.md`.

## 8. Challenges Encountered

- **The Llama/Qwen training-speed reversal (Section 3).** The single most
  time-consuming challenge in this project was chasing down why Llama
  trained 18x slower than Qwen, ruling out tokenizer fragmentation as the
  cause, and eventually root-causing it to an environment-specific
  CPU-offload/gated-repo interaction rather than a real property of either
  model. This also taught me not to trust a single timing anomaly as a
  permanent verdict without instrumenting the actual bottleneck first.
- **Tokenizer fragmentation of Devanagari.** Measured Sanskrit costing 2.45x
  (Llama-3.2) to 4.02x (Qwen2.5) more tokens-per-character than English of
  similar content. This matters because it shrinks the effective context
  window available for Sanskrit content and means the model needs
  proportionally more Sanskrit training tokens to get the same gradient
  signal an equivalent amount of English text would provide — a strong
  argument for tokenizer adaptation as a high-leverage next step, which I
  did not attempt here (explicitly marked optional in the brief) given the
  time budget.
- **`AttributeError` on `model.hf_device_map`** when a model fit entirely on
  a single GPU (the attribute is only set by `accelerate` when a model is
  actually split across devices). Fixed with `getattr(model,
  "hf_device_map", None)` and a fallback device check via
  `next(model.parameters()).device`.
- **Scarcity of instruction-format Sanskrit data.** Good parallel *sentence*
  corpora exist (itihasa); good *instruction* data (varied task phrasings,
  explanation/commentary pairs) essentially doesn't at scale. I addressed
  this by synthetically deriving multiple instruction types from the same
  parallel data (Section 2), with the explicit tradeoff that
  "explanation"/"summarization" targets are proxies, not independently
  authored commentary.
- **Gated model access.** Llama-3.2 requires accepting a license and an HF
  token; I mitigated this by wiring in `Qwen2.5-1.5B-Instruct` as an ungated
  one-line fallback so the pipeline isn't blocked on approval delays.
- **Inference over the full 864-example test set was projected to exceed a
  Colab session's length**, extrapolated from early progress during a run.
  I addressed this by capping the evaluated test set to 100 examples rather
  than rewriting inference to be batched — a scope-reduction tradeoff, not
  a performance fix, and I'm documenting it plainly as such.
- **Initial itihasa parquet load failed** on the first attempt (a dataset
  loading-script deprecation on the HF side); the revision-fallback path
  recovered automatically and the run proceeded normally.
- **Avoiding cherry-picked report examples.** Partway through writing this
  up, I noticed I could unintentionally bias the "good"/"bad" examples
  section just by which ones I happened to scroll past first. I addressed
  this by building `score_and_rank()` — a straightforward sentence-level
  chrF++ ranking over every prediction — so the examples in Section 7 are
  selected by a rule, not by my own eye.
- **Consolidating five scripts into one notebook, then removing the scripts
  entirely** (Section 4) removed a recurring class of bugs where one
  script's output format silently didn't match what the next one expected
  — worth naming as a challenge I solved by simplifying the architecture
  rather than adding more code, and one I finished by not leaving a second,
  divergence-prone copy of the pipeline sitting in the repo.

## 9. What I Would Improve With More Time

- **Add a repetition penalty / no-repeat-ngram constraint at generation
  time**, or drop pure greedy decoding for longer outputs. With 41% of
  predictions showing degenerate repetition loops, this is now the single
  cheapest, highest-leverage fix available, and it doesn't require
  retraining anything.
- **Tokenizer adaptation:** given the fragmentation measured in Section 8,
  extending the base tokenizer's vocabulary with Sanskrit-frequent subwords
  (or continued-pretraining the embedding layer on raw Sanskrit text before
  instruction SFT) is likely the next highest-leverage step after fixing
  repetition.
- **Broader, register-diverse Sanskrit sources:** incorporate GRETIL, the
  Digital Corpus of Sanskrit, and AI4Bharat corpora to cover philosophical
  and technical (Ayurveda/grammar) registers beyond itihasa's epic-poetry
  style — directly relevant to BharatiyaGPT's broader IKS scope, not just
  epic narrative.
- **Real commentary/explanation data:** source or license actual traditional
  commentaries (bhashyas) rather than reusing translations as an explanation
  proxy, to genuinely teach interpretive explanation rather than grounding.
- **Human evaluation:** a native Sanskrit speaker's / Sanskrit scholar's
  review of a sample of outputs would be far more informative than
  automatic metrics alone, especially for explanation/QA quality.
- **A larger training run targeting the en2sa weakness and the repetition
  issue together:** use the full ~6,000 prepared pairs and more epochs,
  since both problems are consistent with an undertrained model at only
  500 examples / 2 epochs.
- **Hyperparameter and architecture ablations:** LoRA rank sweep, LoRA vs.
  full fine-tuning comparison, and a head-to-head between Llama-3.2-1B,
  Qwen2.5-1.5B, and Gemma-2-2B on identical data (bonus items from the
  brief) to make an evidence-backed rather than reasoning-only model choice.
- **Quantized deployment benchmarking:** measure inference latency/quality
  tradeoffs at 4-bit/8-bit for a realistic production-serving estimate,
  relevant to ImmverseAI's stated need for models that are "robust,
  scalable, and production ready."

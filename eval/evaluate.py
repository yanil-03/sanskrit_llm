"""
evaluate.py
===========
Computes quantitative metrics (BLEU, chrF++) over prediction files produced by
infer.py, broken down by task_type, and produces a before-vs-after comparison
table (base model vs fine-tuned model) plus a simple heuristic error-flagging
pass for qualitative failure analysis.

Usage:
    python evaluate.py \
        --base_predictions eval/predictions_base.jsonl \
        --finetuned_predictions eval/predictions.jsonl \
        --out eval/report.md

Metrics:
    - BLEU (sacrebleu) — standard for translation tasks (sa2en / en2sa)
    - chrF++ (sacrebleu) — character-level, more forgiving of Sanskrit's rich
      morphology/compounding than word-level BLEU, and more informative for
      short outputs
    - For non-translation task types (explain/qa/summary) we still report
      chrF++ as a rough proxy of lexical overlap, but flag in the report that
      these should be read qualitatively (see REPORT.md section 6 for why
      n-gram metrics are a weak signal for open-ended generation).
"""

import argparse
import json
from collections import defaultdict

import sacrebleu


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def compute_metrics(rows):
    by_type = defaultdict(list)
    for r in rows:
        by_type[r.get("task_type", "unknown")].append(r)

    metrics = {}
    for task_type, items in by_type.items():
        hyps = [it["prediction"] for it in items]
        refs = [it["reference"] for it in items]
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)  # chrF++
        metrics[task_type] = {
            "n": len(items),
            "bleu": round(bleu.score, 2),
            "chrf++": round(chrf.score, 2),
        }
    return metrics


def flag_heuristic_errors(rows):
    """Cheap heuristics to surface candidate failure cases for manual review:
    - empty/near-empty predictions
    - predictions that just echo the input verbatim (copy failure)
    - predictions much longer than reference (rambling/hallucination)
    - predictions containing repeated n-grams (degenerate repetition)
    """
    flagged = []
    for r in rows:
        pred = r["prediction"].strip()
        inp = r.get("input", "").strip()
        ref = r["reference"].strip()
        reasons = []

        if len(pred) < 2:
            reasons.append("empty_or_near_empty")
        if inp and pred == inp:
            reasons.append("echoed_input_verbatim")
        if ref and len(pred) > 3 * max(len(ref), 1):
            reasons.append("much_longer_than_reference")
        words = pred.split()
        if len(words) >= 6:
            trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
            if len(trigrams) > 0 and len(set(trigrams)) < len(trigrams) * 0.6:
                reasons.append("repetitive_degeneration")

        if reasons:
            flagged.append({**r, "flags": reasons})
    return flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_predictions", type=str, default=None,
                     help="Predictions from the un-finetuned base model")
    ap.add_argument("--finetuned_predictions", type=str, required=True)
    ap.add_argument("--out", type=str, default="eval/report.md")
    ap.add_argument("--n_examples", type=int, default=8,
                     help="Number of qualitative before/after examples to include")
    args = ap.parse_args()

    ft_rows = load_jsonl(args.finetuned_predictions)
    ft_metrics = compute_metrics(ft_rows)
    ft_flagged = flag_heuristic_errors(ft_rows)

    base_rows, base_metrics = None, None
    if args.base_predictions:
        base_rows = load_jsonl(args.base_predictions)
        base_metrics = compute_metrics(base_rows)

    lines = ["# Evaluation Report\n"]

    lines.append("## Quantitative Metrics (fine-tuned model)\n")
    lines.append("| Task Type | N | BLEU | chrF++ |")
    lines.append("|---|---|---|---|")
    for t, m in sorted(ft_metrics.items()):
        lines.append(f"| {t} | {m['n']} | {m['bleu']} | {m['chrf++']} |")
    lines.append("")

    if base_metrics:
        lines.append("## Before vs After (base model vs fine-tuned)\n")
        lines.append("| Task Type | BLEU (base) | BLEU (fine-tuned) | Δ BLEU | "
                      "chrF++ (base) | chrF++ (fine-tuned) | Δ chrF++ |")
        lines.append("|---|---|---|---|---|---|---|")
        for t in sorted(ft_metrics.keys()):
            bm = base_metrics.get(t, {"bleu": float("nan"), "chrf++": float("nan")})
            fm = ft_metrics[t]
            d_bleu = fm["bleu"] - bm["bleu"] if bm["bleu"] == bm["bleu"] else float("nan")
            d_chrf = fm["chrf++"] - bm["chrf++"] if bm["chrf++"] == bm["chrf++"] else float("nan")
            lines.append(
                f"| {t} | {bm['bleu']} | {fm['bleu']} | {d_bleu:+.2f} | "
                f"{bm['chrf++']} | {fm['chrf++']} | {d_chrf:+.2f} |"
            )
        lines.append("")

    lines.append(f"## Heuristically Flagged Failure Cases ({len(ft_flagged)} / {len(ft_rows)})\n")
    lines.append("These are candidates for manual review, not confirmed errors.\n")
    for f in ft_flagged[:15]:
        lines.append(f"- **Flags:** {', '.join(f['flags'])} | **Task:** {f['task_type']}")
        lines.append(f"  - Input: `{f['input'][:120]}`")
        lines.append(f"  - Reference: `{f['reference'][:120]}`")
        lines.append(f"  - Prediction: `{f['prediction'][:120]}`")
    lines.append("")

    lines.append(f"## Qualitative Before/After Examples (first {args.n_examples})\n")
    for i in range(min(args.n_examples, len(ft_rows))):
        r = ft_rows[i]
        lines.append(f"**Example {i+1}** ({r['task_type']})")
        lines.append(f"- Instruction: {r['instruction']}")
        lines.append(f"- Input: `{r['input']}`")
        lines.append(f"- Reference: `{r['reference']}`")
        if base_rows and i < len(base_rows):
            lines.append(f"- Base model output: `{base_rows[i]['prediction']}`")
        lines.append(f"- Fine-tuned model output: `{r['prediction']}`")
        lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote report to {args.out}")
    print(json.dumps(ft_metrics, indent=2))


if __name__ == "__main__":
    main()

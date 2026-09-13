"""
inspect_tokenizer.py
=====================
Quick diagnostic: how badly does the base model's tokenizer fragment Sanskrit
(Devanagari) text compared to English text of similar length? This directly
addresses the "Tokenization / Vocabulary" challenge called out in the
assignment brief.

We report:
  - tokens-per-character ratio for Sanskrit vs English samples
  - a visual token breakdown of a few sample sentences
  - how many Sanskrit sentences get split into >2x more tokens than their
    English translation of similar character length (a proxy for how much
    "compute budget" Sanskrit costs relative to English at the same content
    length -- directly relevant to context-window and training-cost planning)

Usage:
    python inspect_tokenizer.py --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import json

from transformers import AutoTokenizer


SAMPLES = [
    ("वसुधैव कुटुम्बकम्।", "The whole world is one family."),
    ("धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।",
     "On the field of dharma, the field of Kuru, assembled and eager to fight."),
    ("कर्मण्येवाधिकारस्ते मा फलेषु कदाचन।",
     "You have a right to perform your duty, but not to the fruits of action."),
    ("सत्यमेव जयते।", "Truth alone triumphs."),
    ("अहिंसा परमो धर्मः।", "Non-violence is the highest duty."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"Tokenizer: {args.model}")
    print(f"Vocab size: {tok.vocab_size}\n")

    report = []
    for sa, en in SAMPLES:
        sa_ids = tok.encode(sa, add_special_tokens=False)
        en_ids = tok.encode(en, add_special_tokens=False)

        sa_tokens = tok.convert_ids_to_tokens(sa_ids)
        en_tokens = tok.convert_ids_to_tokens(en_ids)

        sa_tpc = len(sa_ids) / max(len(sa), 1)
        en_tpc = len(en_ids) / max(len(en), 1)

        print(f"SA: {sa}")
        print(f"  chars={len(sa)}  tokens={len(sa_ids)}  tokens/char={sa_tpc:.2f}")
        print(f"  breakdown: {sa_tokens}")
        print(f"EN: {en}")
        print(f"  chars={len(en)}  tokens={len(en_ids)}  tokens/char={en_tpc:.2f}")
        print(f"  breakdown: {en_tokens}")
        print(f"  --> Sanskrit costs {sa_tpc / en_tpc:.2f}x more tokens/char than English\n")

        report.append({
            "sanskrit": sa, "english": en,
            "sa_chars": len(sa), "sa_tokens": len(sa_ids), "sa_tokens_per_char": round(sa_tpc, 3),
            "en_chars": len(en), "en_tokens": len(en_ids), "en_tokens_per_char": round(en_tpc, 3),
            "ratio_sa_over_en": round(sa_tpc / en_tpc, 2),
        })

    avg_ratio = sum(r["ratio_sa_over_en"] for r in report) / len(report)
    print(f"Average Sanskrit/English tokens-per-char ratio: {avg_ratio:.2f}x")
    print("\nInterpretation: if this ratio is well above 1.0, the tokenizer was\n"
          "not trained with meaningful Devanagari/Sanskrit representation, and\n"
          "Sanskrit text consumes a disproportionate share of the context\n"
          "window and gradient signal per unit of content relative to English.\n"
          "See REPORT.md for whether we attempted tokenizer adaptation and why.")

    with open("eval/tokenizer_report.json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "samples": report, "avg_ratio": avg_ratio}, f,
                   ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

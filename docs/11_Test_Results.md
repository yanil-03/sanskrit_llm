# 11. Test Results — Success & Failure Cases

All examples below are pulled via `score_and_rank()` (sentence-level chrF++ across the full 100-example test set) — not hand-picked — using the current default model, **Llama-3.2-1B-Instruct**.

## 11.1 Successful Cases (top-ranked by chrF++, across task types)

| chrF++ | Task | Input | Reference | Prediction |
|---|---|---|---|---|
| 45.5 | explain | ततस्तत्र प्रविष्टस्य कौसल्याया निवेशनम्... | Having entered Kausalyā's apartment, the king... was overwhelmed with emotion. | And thereupon, the king, having entered the abode of Kausalyā, was struck with grief, and was overcome with sorrow. |
| 34.0 | qa | देवैस्तदा समागम्य सर्षिसङ्घः सचारणैः... | Upon the assembled gods with the saints and the Cāraṇas beseeching those two foremost of celestials, they became pacified. | And thereupon the celestials came to the assembly of the celestials, and the mighty ones came to the assembly of the gods. |
| 32.9 | explain | कल्यमुत्थाय देवानां कृत्वा पूजां यथाविधि... | Rising from the bed early in the morning, adore the deities... | And when the celestials had performed the worship, I, Dāśaratha, was delighted to hear the words of my father. |
| 27.7 | sa2en | सुकुमारं च बालं च सततं लालितं त्वया... | O father, where have you gone, leaving the tender and youthful Bharata fondled by you, to lament (your loss). | O you, O Bharata, you are the son of Kumaara, the daughter of Lala, and you are the one who has been blessed with the boon of being able to be killed by the bow. |

> Takeaway: "explain" is consistently the strongest task — entities and topic are captured correctly, and prose stays fluent, even though specific details drift (e.g. "emotion" → "grief and sorrow"). The qa and sa2en top examples show the model correctly picking up character names and general register, even when the literal content wanders.

## 11.2 Failure Cases (bottom-ranked by chrF++) — repetitive degeneration is now the dominant pattern

| chrF++ | Task | Input | Reference | Prediction |
|---|---|---|---|---|
| 4.4 | sa2en | अनेन धनुषा राम हत्वा संख्ये महासुरान्... | Having, O Rama, slain the mighty Asuras with this bow... | O Rāma, O Rāma, O Rāma, O Rāma, O Rāma, O Rāma... *(repeats ~25 more times)* |
| 4.2 | en2sa | And coming before that best of regenerate ones, he saw that sage's son near Romapāda... | आसाद्य तं द्विजश्रेष्ठं रोमपादसमीपगम्... | स ते रमपादा त्वं त्वं त्वं त्वं त्वं... *(repeats "त्वं" ~40 times)* |
| 3.1 | en2sa | Hearing my footsteps, the ascetic said, "Why, my son, delay you? Bring the drink at once. | पदशब्दं तु मे श्रुत्वा मुनिर्वाक्यमभाषत... | न त्वं त्वं त्वं त्वं त्वं... *(repeats "त्वं" for the rest of the output)* |
| 2.6 | qa | एष राम शिवः पन्था यत्रैते पुष्पिता द्रुमाः... | This is the way, O Rāma, leading to the mount Rsyamūka... | O Rāma, O Sītā, O Sītā, O Sītā, O Sītā... *(repeats ~30 times)* |

> Takeaway: this is a meaningfully different failure mode than what I saw with Qwen. Qwen's en2sa failures were coherent-but-wrong Sanskrit (semantic hallucination). Llama's failures are **degenerate repetition loops** — the model gets a plausible start, then locks onto a token/phrase and repeats it until it hits the token limit. The heuristic flagging pass caught this in **41 of 100 predictions (41%)**, spread across every task type, not just en2sa. This points to a decoding-side fix (repetition penalty, no-repeat-ngram, or a lower `max_new_tokens` for short source verses) as the highest-leverage next step, separate from any data or training changes.

## 11.3 Per-Task-Type Breakdown (top/bottom, from the full ranking)

**en2sa** — top example still isn't a clean translation (chrF++ 20.1: input about Kausalyā's forgiveness → output roughly grammatical but not matching the reference), and the task remains the weakest overall (chrF++ 10.21 average). All 3 bottom en2sa examples are pure repetition loops.

**explain** — clear strongest task type (chrF++ 21.35 average, up to 45.5 on the best example). Even its worst examples (chrF++ 12–14) are still coherent English, just factually adrift, rather than degenerate.

**qa / summary** — smallest sample sizes (N=10, N=6) and the only two task types where chrF++ regressed after fine-tuning (see [10_Evaluation_Metrics.md](./10_Evaluation_Metrics.md)); several of their worst examples are also repetition loops rather than wrong-but-fluent answers.

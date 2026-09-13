# 4. How the Data Is Structured

There's no database here — this is a one-shot fine-tuning job, so plain JSONL files did the job fine. Here's how it's laid out.

## 4.1 What One Training Example Looks Like
Every train/val/test example is a single JSON object, one per line:

```json
{
  "instruction": "Translate the following Sanskrit text into English.",
  "input": "वसुधैव कुटुम्बकम्।",
  "output": "The whole world is one family.",
  "task_type": "sa2en"
}
```

## 4.2 Field by Field

| Field | Type | What it holds |
|---|---|---|
| instruction | string | The prompt shown to the model (I randomly sample from a few phrasings per task) |
| input | string | The source text — Sanskrit or English, depending on the task |
| output | string | The target text the model is supposed to produce |
| task_type | enum | One of: `sa2en`, `en2sa`, `explain`, `qa`, `summary` — lets me break metrics down by task |

## 4.3 How I Split the Data

| Split | Where it came from | What it's for |
|---|---|---|
| train.jsonl | 90% of the expanded itihasa pairs | Actual training |
| val.jsonl | 5% of the expanded itihasa pairs | Watching validation loss during training |
| test.jsonl | 5% of the expanded itihasa pairs + all of my hand-picked seed verses | Held-out evaluation — the seed verses are famous, easy-to-verify ones (Gita lines, well-known subhashitas) so I can sanity-check things by eye |

## 4.4 How One Pair Becomes Several Training Examples
Every raw (Sanskrit, English) pair from itihasa gets expanded into 2-5 instruction examples, using a simple modulo rule on the pair's index — so I get a mix of task types without having to hand-author separate data for each one:

| task_type | Made from | Where the target text comes from |
|---|---|---|
| sa2en | Every pair | The original English translation |
| en2sa | Every pair | The original Sanskrit text |
| explain | Every 3rd pair | The English translation, with "This verse conveys:" tacked on the front |
| qa | Every 4th pair | The original English translation |
| summary | Every 5th pair, if the English is more than 6 words | The original English translation |

Worth being upfront about: "explain" and "summary" aren't independently written commentary — they're the same translation reused with a different framing. I didn't have a licensed Sanskrit commentary corpus available in this timeframe, so this was the practical shortcut. I flag this explicitly rather than pretending otherwise.

## 4.5 What Comes Out the Other End (outputs/)
- `adapter_model.safetensors` — the trained LoRA weights (small — a few hundred MB at most, not a full model copy)
- `adapter_config.json` — the LoRA hyperparameters (rank, alpha, which layers it's attached to)
- `run_config.json` — the exact CONFIG dict used for that run, so I (or anyone else) can reproduce it later
- tokenizer files — saved alongside the adapter so it can be loaded on its own, without needing anything else

"""
rephrase_few_shots.py — Rewrite few-shot traces in retail_norms_few_shots.json
with synthetic customer data, preserving structure and verdict-relevant behavior.

Usage:
    OPENAI_API_KEY=sk-... python rephrase_few_shots.py
    # or with a different model:
    OPENAI_API_KEY=... python rephrase_few_shots.py --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import litellm

HERE = Path(__file__).resolve().parent
FEW_SHOTS_PATH = HERE / "retail_norms_few_shots.json"


_REPHRASE_SYSTEM = (
    "You are a precise data-transformation assistant. "
    "You rewrite conversation traces to use synthetic data while preserving "
    "structure, tool-call names, and behavioral patterns exactly."
)

_REPHRASE_PROMPT = """\
Rewrite the customer-service conversation below so it uses completely different \
fictional customer data, while preserving:

1. The exact same line format: [N] role: content
2. The exact same TOOL_CALL function names (only change argument values)
3. The same sequence and number of turns
4. The same behavioral pattern — the verdict for this example is **{verdict}** \
and must remain so in the rewrite

Change everything else:
- Customer name, email, zip → a different, clearly fictional person
- Order IDs → new made-up IDs (format: #X followed by 7 digits, e.g. #B3847291)
- User IDs → derived from the new fictional name (e.g. jane_doe_5821)
- Product names, descriptions, prices → different plausible retail items
- Tool argument values and tool result JSON → updated to reflect the new data
- Message wording → rephrased (same intent, different words)

Output ONLY the rewritten trace, nothing else — no commentary, no labels, \
no markdown fences.

--- Original trace ---
{trace}
"""


async def rephrase_one(
    norm: str,
    idx: int,
    example: dict,
    model: str,
    api_key: str,
    sem: asyncio.Semaphore,
) -> dict:
    prompt = _REPHRASE_PROMPT.format(
        verdict=example["verdict"],
        trace=example["trace"],
    )
    async with sem:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": _REPHRASE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            api_key=api_key,
        )
    rephrased = response.choices[0].message.content.strip()
    print(f"  ✓  {norm}[{idx}]  ({len(rephrased)} chars)", flush=True)
    return {"trace": rephrased, "verdict": example["verdict"]}


async def main(model: str, concurrency: int) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    with open(FEW_SHOTS_PATH) as f:
        data: dict = json.load(f)

    sem = asyncio.Semaphore(concurrency)
    tasks = []
    keys = []  # (norm, idx)

    for norm, entry in data.items():
        for idx, example in enumerate(entry["few_shots"]):
            tasks.append(rephrase_one(norm, idx, example, model, api_key, sem))
            keys.append((norm, idx))

    print(f"Rephrasing {len(tasks)} examples with {model} (concurrency={concurrency}) …")
    results = await asyncio.gather(*tasks)

    # Write back in place
    for (norm, idx), rephrased in zip(keys, results):
        data[norm]["few_shots"][idx] = rephrased

    with open(FEW_SHOTS_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nDone — wrote {len(tasks)} rephrased examples to {FEW_SHOTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--concurrency", "-c", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(main(args.model, args.concurrency))

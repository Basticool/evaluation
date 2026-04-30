"""
create_judge_few_shots.py — Generate retail_norms_few_shots.json.

Reads results/results.jsonl (NormMonitor GT outcomes) and
merged_traces_labels.jsonl (full conversations), then selects
up to 2 satisfied + 1 violated example per norm as LLM judge few-shots.
"""

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

_DEFAULT_RESULTS  = HERE / "results" / "results.jsonl"
_DEFAULT_DATASET  = HERE / "merged_traces_labels.jsonl"
_DEFAULT_OUT      = HERE / "retail_norms_few_shots.json"


def _tc_name(tc: dict) -> str:
    fn = tc.get("function") or {}
    return fn.get("name") or tc.get("name") or ""


def _tc_args(tc: dict) -> dict:
    fn = tc.get("function") or {}
    raw = fn.get("arguments") or tc.get("arguments") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    return raw if isinstance(raw, dict) else {}


def format_trace(messages: list[dict]) -> str:
    lines = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content") or ""
        tcs = msg.get("tool_calls") or []

        if role == "system":
            lines.append(f"[{i}] system: {content[:200]}")
            continue

        if role == "assistant" and tcs:
            tc_parts = []
            for tc in tcs:
                name = _tc_name(tc)
                args = _tc_args(tc)
                tc_parts.append(f"TOOL_CALL {name}({json.dumps(args, ensure_ascii=False)})")
            joined = "\n    ".join(tc_parts)
            prefix = f"{content}\n    " if content.strip() else ""
            lines.append(f"[{i}] assistant: {prefix}{joined}")
        elif role == "tool":
            lines.append(f"[{i}] tool_result: {content}")
        else:
            lines.append(f"[{i}] {role}: {content}")

    return "\n".join(lines)


def main() -> None:
    results: list[dict] = []
    with open(_DEFAULT_RESULTS) as f:
        for line in f:
            results.append(json.loads(line))

    dataset: list[dict] = []
    with open(_DEFAULT_DATASET) as f:
        for line in f:
            dataset.append(json.loads(line))

    # Group results by norm, then by GT outcome
    by_norm: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_norm[r["norm"]][r["gt"]].append(r)

    few_shots: dict[str, dict] = {}
    for norm, by_outcome in sorted(by_norm.items()):
        satisfied = by_outcome.get("satisfied", [])[:2]
        violated  = by_outcome.get("violated",  [])[:1]
        examples  = []
        for r in satisfied:
            msgs = dataset[r["orig_idx"]]["simulation"]["messages"]
            examples.append({"trace": format_trace(msgs), "verdict": "satisfied"})
        for r in violated:
            msgs = dataset[r["orig_idx"]]["simulation"]["messages"]
            examples.append({"trace": format_trace(msgs), "verdict": "violated"})
        if examples:
            few_shots[norm] = {"few_shots": examples}

    with open(_DEFAULT_OUT, "w") as f:
        json.dump(few_shots, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(few_shots)} norms to {_DEFAULT_OUT}")
    for norm, data in sorted(few_shots.items()):
        counts: dict[str, int] = defaultdict(int)
        for ex in data["few_shots"]:
            counts[ex["verdict"]] += 1
        print(f"  {norm:<50} {dict(counts)}")


if __name__ == "__main__":
    main()

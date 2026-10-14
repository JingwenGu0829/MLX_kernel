"""Evaluate a candidate kernel file against a task.

    uv run python -m kbench.run add_rmsnorm candidates/add_rmsnorm_hand.py

A candidate is a Python file that defines `forward(*inputs)` with the same
signature and outputs as the task's reference.
"""

import argparse
import importlib.util
import json
from pathlib import Path

from kbench.harness import evaluate, format_report
from kbench.tasks import TASKS


def load_forward(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("candidate")
    ap.add_argument("--train-only", action="store_true", help="skip held-out shapes")
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args()

    splits = ("train",) if args.train_only else ("train", "heldout")
    report = evaluate(TASKS[args.task], load_forward(args.candidate), splits)
    report["candidate"] = args.candidate
    print(format_report(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

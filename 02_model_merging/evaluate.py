"""Accuracy of one model on every task.

    python evaluate.py                      # the base model
    python evaluate.py runs/sst2            # a fine-tuned or merged checkpoint
"""

import argparse
import json
import time

from common import BASE_MODEL, TASKS, accuracy, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default=BASE_MODEL)
    parser.add_argument("-n", type=int, default=500, help="examples per task")
    args = parser.parse_args()

    model, tokenizer, device = load(args.model)
    scores = {}
    for task in TASKS.values():
        started = time.time()
        scores[task.name] = accuracy(model, tokenizer, device, task, n=args.n)
        chance = 1 / len(task.labels)
        print(
            f"{task.name}: {scores[task.name]:.1%} "
            f"(chance {chance:.0%}, n={args.n}, {time.time() - started:.0f}s, {device})"
        )
    print(json.dumps({"model": args.model, "n": args.n, **scores}))


if __name__ == "__main__":
    main()

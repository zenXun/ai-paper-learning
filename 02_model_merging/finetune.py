"""Full fine-tuning of the base model on one task.

    python finetune.py sst2
    python finetune.py ag_news

All weights are updated (no LoRA), so `finetuned - base` is a task vector in
the sense of the Task Arithmetic paper. The loss is cross-entropy on the single
answer token that follows the prompt.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import TASKS, accuracy, examples, label_token_ids, load, prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--mem-limit-gb", type=float, default=9.0, help="abort before the machine swaps"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    task = TASKS[args.task]
    model, tokenizer, device = load()
    label_ids = torch.tensor(label_token_ids(tokenizer, task), device=device)
    data = examples(task, "train", args.n_train, seed=args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    model.train()
    started = time.time()
    steps = 0
    for start in range(0, len(data), args.batch_size):
        batch = data[start : start + args.batch_size]
        prompts = [prompt(tokenizer, task, text) for text in batch[task.text_field]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        target = label_ids[torch.tensor(batch["label"], device=device)]
        loss = F.cross_entropy(model(**enc).logits[:, -1, :], target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        steps += 1
        if device == "mps":
            torch.mps.empty_cache()  # batches vary in length; cached buffers pile up otherwise
            used = torch.mps.driver_allocated_memory() / 2**30
            if used > args.mem_limit_gb:
                raise SystemExit(f"step {steps}: {used:.1f}GB allocated, over the limit")
        if steps % 10 == 0:
            line = f"step {steps} loss {loss.item():.3f} {(time.time() - started) / steps:.2f}s/step"
            if device == "mps":
                line += f" mem {torch.mps.driver_allocated_memory() / 2**30:.1f}GB"
            print(line, flush=True)
        if args.max_steps and steps >= args.max_steps:
            break

    scores = {t.name: accuracy(model, tokenizer, device, t) for t in TASKS.values()}
    print(json.dumps({"task": task.name, "steps": steps, "lr": args.lr, **scores}))
    if not args.no_save:
        out = Path("runs") / task.name
        model.save_pretrained(out)
        (out / "scores.json").write_text(json.dumps(scores))
        print(f"saved {out}")


if __name__ == "__main__":
    main()

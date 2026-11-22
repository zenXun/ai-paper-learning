# 02 — Model merging

Fine-tune one small base model on two tasks, then combine the two results by
operating on the weights directly, with no further training. The module follows
the repo method: write each merge from its equations first, then compare with a
mature implementation ([mergekit](https://github.com/arcee-ai/mergekit)).

Status: environment, baseline and fine-tuning script are in place. The two
fine-tuned checkpoints and every merge are still to do.

## Papers

| Step | Paper | Claim to check here |
|---|---|---|
| 1 | Model Soups (Wortsman et al. 2022) | Averaging two fine-tunes of one base keeps both tasks |
| 2 | Task Arithmetic (Ilharco et al. 2023) | Task vectors of different tasks are close to orthogonal |
| 3 | TIES-Merging (Yadav et al. 2023) | Trimming and sign election beat plain addition |
| 4 | DARE (Yu et al. 2024) | Most of a task vector can be dropped at random |

## Setup

mergekit needs Python 3.10 or newer.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Layout

| File | What it does |
|---|---|
| `common.py` | Base model, the two tasks, prompt format, accuracy metric |
| `evaluate.py` | Accuracy of one model on both tasks |
| `finetune.py` | Full fine-tuning on one task, saves to `runs/<task>` (not in git) |

Base model: `HuggingFaceTB/SmolLM2-360M-Instruct` (362M parameters, Apache-2.0).

Tasks: SST-2 (sentiment, 2 labels) and AG News (topic, 4 labels). Each is
phrased as a chat turn that the model answers with one label word. Accuracy
compares the logits of the label words at the first answer position, over 500
examples of the evaluation split.

## Measured so far

Apple M2, 16 GB, PyTorch on MPS, 2026-11-22.

| Model | SST-2 | AG News |
|---|---|---|
| Chance | 50.0% | 25.0% |
| Base, no fine-tuning | 50.4% | 44.8% |
| 40 steps on AG News (smoke test, not saved) | 60.2% | 81.8% |

The base model answers "negative" for almost every SST-2 review (199 of 200 in
a spot check), which is why it sits at chance.

Full fine-tuning fits on this machine at batch size 4: 6.1 GB allocated, about
1.6 s per step. At batch size 8 with inputs up to 384 tokens the process grew
past 11 GB and the machine started swapping, so `finetune.py` clips inputs to
128 tokens, clears the MPS cache every step and aborts above 9 GB.

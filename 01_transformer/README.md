# 01 — Transformer / Attention

> Paper: [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
> Status: **in progress** — paper notes phase

## What this module is

A from-scratch, minimal **decoder-only** Transformer implemented in PyTorch and trained on a character-level language modeling toy task. The goal is to internalize the design choices in the original paper, not to be fast or production-ready.

## Planned structure

```
01_transformer/
├── README.md            # this file
├── notes/               # paper reading notes (why each design choice exists)
├── model.py             # minimal decoder-only Transformer (TBD)
├── train.py             # character-level LM training loop (TBD)
├── visualize.py         # attention heatmaps, positional encoding plots (TBD)
└── notebooks/           # exploratory analysis (TBD)
```

## How to run

> _Coming soon — the code will land after the paper-reading phase._

## Design choices to be discussed

These are the questions the implementation and write-up will answer:

- Why scaled dot-product attention? Why the `√dk` scaling specifically?
- Why multi-head, and what do different heads actually learn?
- Why sinusoidal positional encoding vs. learned vs. RoPE?
- Pre-LN vs. Post-LN — what changed and why does it matter for training stability?
- Why residual connections everywhere?

## References

- Original paper: <https://arxiv.org/abs/1706.03762>
- (After self-implementation) comparison reference: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

## Companion blog post

To be published at [zenxhunger.com](https://zenxhunger.com) once this module is complete.

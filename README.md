# ai_research

A learning repo: read AI papers → implement minimal MVPs → distill into tutorials.

Each module under this repo corresponds to one paper / one concept. Long-form write-ups live at [zenxhunger.com](https://zenxhunger.com); this repo holds the code, with each module's `README.md` acting as its self-contained "how to run / how it's built" guide.

## Methodology

For every module:

1. **Read the paper.** Focus on *why* each design choice was made, not just *what* it is.
2. **Implement a minimal MVP.** From the equations, not by copying a reference. Optimize for readability.
3. **Compare.** Once the MVP works, diff against a mature reference implementation (e.g. nanoGPT). Differences are themselves teaching material.
4. **Visualize.** Produce assets that make the mechanism legible (attention maps, loss curves, etc.).
5. **Write it up.** A self-contained module `README.md` (this repo) + a narrative blog post ([zenxhunger.com](https://zenxhunger.com)).

## Modules

| # | Topic | Paper | Status |
|---|-------|-------|--------|
| 01 | Transformer / Attention | [Attention Is All You Need (Vaswani et al. 2017)](https://arxiv.org/abs/1706.03762) | in progress |
| 02 | Model merging | [Model Soups](https://arxiv.org/abs/2203.05482), [Task Arithmetic](https://arxiv.org/abs/2212.04089), [TIES](https://arxiv.org/abs/2306.01708), [DARE](https://arxiv.org/abs/2311.03099) | in progress |

## Layout

```
ai_research/
├── README.md            # this file
├── 01_transformer/
│   └── README.md        # module entry: setup, train, visualize
└── ...                  # future modules
```

## Tech baseline

- **Language model architecture**: decoder-only by default (GPT-style)
- **Default toy task**: character-level language modeling
- **Stack**: Python + PyTorch, kept deliberately small per module

"""Hand-written model merging algorithms in JAX (PyTree + jnp + explicit PRNG).

Run step by step to inspect each paper's claim on our two fine-tuned checkpoints
(`runs/sst2` and `runs/ag_news`):

    .venv/bin/python merge_jax.py cosine
    .venv/bin/python merge_jax.py linear --alpha 0.5
    .venv/bin/python merge_jax.py arithmetic --lam 0.5 0.7 1.0
    .venv/bin/python merge_jax.py ties --density 0.2 --lam 0.5 0.7 1.0
    .venv/bin/python merge_jax.py dare --drop 0.0 0.5 0.8 0.9 0.95 0.99 --lam 0.5
"""

import argparse
import gc
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from common import BASE_MODEL, TASKS, accuracy, load, pick_device


# ---------- PyTorch <-> JAX PyTree bridge (kept on CPU to stay under 16GB) ----------


def load_weights_jax(model_path: str) -> dict[str, jax.Array]:
    """Load a Hugging Face checkpoint into a JAX PyTree (dict of float32 arrays on CPU)."""
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    weights = {k: jnp.asarray(v.detach().cpu().numpy()) for k, v in model.state_dict().items()}
    del model
    gc.collect()
    return weights


def evaluate_weights(weights_jax: dict[str, jax.Array], n: int = 500) -> dict[str, float]:
    """Load merged JAX weights back into a PyTorch model and score both tasks on MPS/CPU."""
    device = pick_device()
    model, tokenizer, device = load(BASE_MODEL, device="cpu")
    state_dict = {k: torch.from_numpy(np.array(v, copy=True)) for k, v in weights_jax.items()}
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model = model.to(device)

    scores = {}
    for task in TASKS.values():
        scores[task.name] = accuracy(model, tokenizer, device, task, n=n)

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return scores


# ---------- Core JAX merging equations ----------


def task_vector(w_ft: dict[str, jax.Array], w_base: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """tau = theta_ft - theta_base (one-line PyTree subtraction)."""
    return jax.tree.map(lambda ft, b: ft - b, w_ft, w_base)


def cosine_similarity_tree(tau_a: dict[str, jax.Array], tau_b: dict[str, jax.Array]) -> dict[str, float]:
    """Global cosine similarity across all 362M parameters without concatenating into one giant array."""
    dot = sum(float(jnp.sum(x)) for x in jax.tree.leaves(jax.tree.map(lambda a, b: a * b, tau_a, tau_b)))
    norm_a = sum(float(jnp.sum(x)) for x in jax.tree.leaves(jax.tree.map(lambda a: a * a, tau_a))) ** 0.5
    norm_b = sum(float(jnp.sum(x)) for x in jax.tree.leaves(jax.tree.map(lambda b: b * b, tau_b))) ** 0.5
    return {
        "cosine": dot / (norm_a * norm_b + 1e-12),
        "l2_norm_a": norm_a,
        "l2_norm_b": norm_b,
    }


def linear_merge(w_a: dict[str, jax.Array], w_b: dict[str, jax.Array], alpha: float = 0.5) -> dict[str, jax.Array]:
    """Model Soups / Linear average: alpha * theta_A + (1 - alpha) * theta_B."""
    return jax.tree.map(lambda a, b: alpha * a + (1.0 - alpha) * b, w_a, w_b)


def task_arithmetic(
    w_base: dict[str, jax.Array], taus: list[dict[str, jax.Array]], lam: float = 0.5
) -> dict[str, jax.Array]:
    """Task Arithmetic: theta_base + lambda * sum(tau_i)."""
    return jax.tree.map(lambda b, *ts: b + lam * sum(ts), w_base, *taus)


def _trim_leaf(t: jax.Array, density: float) -> jax.Array:
    """Step 1 of TIES: keep top `density` fraction by magnitude, zero out the rest."""
    if density >= 1.0:
        return t
    abs_t = jnp.abs(t)
    # quantile(1 - density) gives the cutoff above which `density` of entries lie
    threshold = jnp.quantile(abs_t, 1.0 - density)
    return jnp.where(abs_t >= threshold, t, 0.0)


def ties_merge(
    w_base: dict[str, jax.Array],
    taus: list[dict[str, jax.Array]],
    density: float = 0.2,
    lam: float = 0.7,
) -> dict[str, jax.Array]:
    """TIES-Merging: 1) Trim -> 2) Elect Sign -> 3) Disjoint Mean -> 4) Scale."""

    def merge_leaf(b, *ts):
        # 1. Trim small redundant updates per task
        trimmed = [_trim_leaf(t, density) for t in ts]
        # 2. Elect Sign: total positive magnitude vs total negative magnitude = sign(sum)
        elected = jnp.sign(sum(trimmed))
        # 3. Disjoint Mean: average only values whose sign matches the elected sign
        aligned = [jnp.where((jnp.sign(t) == elected) & (t != 0.0), t, 0.0) for t in trimmed]
        counts = sum((jnp.sign(t) == elected) & (t != 0.0) for t in trimmed)
        disjoint_mean = jnp.where(counts > 0, sum(aligned) / jnp.maximum(counts, 1.0), 0.0)
        # 4. Scale and add back to base
        return b + lam * disjoint_mean

    return jax.tree.map(merge_leaf, w_base, *taus)


def dare_sparsify(
    tau: dict[str, jax.Array], drop_rate: float, key: jax.Array
) -> dict[str, jax.Array]:
    """DARE: randomly zero out `drop_rate` of delta weights and rescale the rest by 1/(1 - p).

    Demonstrates JAX's stateless PRNG: split one parent key into one unique subkey per tensor.
    """
    if drop_rate <= 0.0:
        return tau
    keep_prob = 1.0 - drop_rate
    leaves, treedef = jax.tree.flatten(tau)
    subkeys = jax.random.split(key, len(leaves))
    new_leaves = [
        jnp.where(jax.random.bernoulli(sk, keep_prob, shape=leaf.shape), leaf / keep_prob, 0.0)
        for leaf, sk in zip(leaves, subkeys)
    ]
    return jax.tree.unflatten(treedef, new_leaves)


# ---------- CLI ----------


def main():
    parser = argparse.ArgumentParser(description="Hand-written JAX model merging experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("cosine", help="Measure cosine similarity & L2 norms of the two task vectors")

    p_lin = sub.add_parser("linear", help="Linear / Model Soups average")
    p_lin.add_argument("--alpha", type=float, nargs="+", default=[0.5])
    p_lin.add_argument("-n", type=int, default=500)

    p_ta = sub.add_parser("arithmetic", help="Task Arithmetic: base + lam * (tau_a + tau_b)")
    p_ta.add_argument("--lam", type=float, nargs="+", default=[0.5, 0.7, 1.0])
    p_ta.add_argument("-n", type=int, default=500)

    p_ties = sub.add_parser("ties", help="TIES-Merging: Trim -> Elect -> Disjoint Mean -> Scale")
    p_ties.add_argument("--density", type=float, default=0.2)
    p_ties.add_argument("--lam", type=float, nargs="+", default=[0.5, 0.7, 1.0])
    p_ties.add_argument("-n", type=int, default=500)

    p_dare = sub.add_parser("dare", help="DARE + Task Arithmetic sweep over drop rates")
    p_dare.add_argument("--drop", type=float, nargs="+", default=[0.0, 0.5, 0.8, 0.9, 0.95, 0.99])
    p_dare.add_argument("--lam", type=float, default=0.5)
    p_dare.add_argument("--seed", type=int, default=0)
    p_dare.add_argument("-n", type=int, default=500)

    args = parser.parse_args()

    t0 = time.time()
    print("Loading checkpoints into JAX PyTrees (CPU)...", flush=True)
    w_sst2 = load_weights_jax("runs/sst2")
    w_ag = load_weights_jax("runs/ag_news")

    if args.cmd == "linear":
        for a in args.alpha:
            merged = linear_merge(w_sst2, w_ag, alpha=a)
            scores = evaluate_weights(merged, n=args.n)
            avg = (scores["sst2"] + scores["ag_news"]) / 2
            print(json.dumps({"method": "linear", "alpha": a, **scores, "avg": round(avg, 4)}), flush=True)
        return

    w_base = load_weights_jax(BASE_MODEL)
    tau_sst2 = task_vector(w_sst2, w_base)
    tau_ag = task_vector(w_ag, w_base)
    del w_sst2, w_ag
    gc.collect()
    print(f"Prepared task vectors in {time.time() - t0:.1f}s.", flush=True)

    if args.cmd == "cosine":
        stats = cosine_similarity_tree(tau_sst2, tau_ag)
        print(json.dumps({"method": "cosine", **stats}, indent=2))
    elif args.cmd == "arithmetic":
        for lam in args.lam:
            merged = task_arithmetic(w_base, [tau_sst2, tau_ag], lam=lam)
            scores = evaluate_weights(merged, n=args.n)
            avg = (scores["sst2"] + scores["ag_news"]) / 2
            print(json.dumps({"method": "task_arithmetic", "lam": lam, **scores, "avg": round(avg, 4)}), flush=True)
    elif args.cmd == "ties":
        for lam in args.lam:
            merged = ties_merge(w_base, [tau_sst2, tau_ag], density=args.density, lam=lam)
            scores = evaluate_weights(merged, n=args.n)
            avg = (scores["sst2"] + scores["ag_news"]) / 2
            print(
                json.dumps(
                    {"method": "ties", "density": args.density, "lam": lam, **scores, "avg": round(avg, 4)}
                ),
                flush=True,
            )
    elif args.cmd == "dare":
        base_key = jax.random.PRNGKey(args.seed)
        for p in args.drop:
            k1, k2 = jax.random.split(base_key, 2)
            t1 = dare_sparsify(tau_sst2, p, k1)
            t2 = dare_sparsify(tau_ag, p, k2)
            merged = task_arithmetic(w_base, [t1, t2], lam=args.lam)
            scores = evaluate_weights(merged, n=args.n)
            avg = (scores["sst2"] + scores["ag_news"]) / 2
            print(
                json.dumps({"method": "dare", "drop": p, "lam": args.lam, **scores, "avg": round(avg, 4)}),
                flush=True,
            )


if __name__ == "__main__":
    main()

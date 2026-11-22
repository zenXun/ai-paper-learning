"""Shared pieces for the model-merging experiments: tasks, prompts, scoring.

Every task is a classification task phrased as chat: the user message holds the
text, the assistant answers with one label word. We score a model by looking at
the logits of the first token of each label word right after the prompt, so the
metric is plain accuracy and needs no text generation.
"""

from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
MAX_TEXT_TOKENS = 128  # longer inputs are cut here, before the chat template is applied


@dataclass(frozen=True)
class Task:
    name: str
    dataset: str
    text_field: str
    eval_split: str
    labels: tuple[str, ...]  # index = dataset label id
    instruction: str


TASKS = {
    "sst2": Task(
        name="sst2",
        dataset="stanfordnlp/sst2",
        text_field="sentence",
        eval_split="validation",
        labels=("negative", "positive"),
        instruction="Is the sentiment of this movie review negative or positive? "
        "Answer with one word.",
    ),
    "ag_news": Task(
        name="ag_news",
        dataset="fancyzhx/ag_news",
        text_field="text",
        eval_split="test",
        labels=("World", "Sports", "Business", "Technology"),
        instruction="Which topic is this news article about: World, Sports, "
        "Business or Technology? Answer with one word.",
    ),
}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(model_path: str = BASE_MODEL, device: str | None = None):
    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"  # the answer position must be the last token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    return model.to(device), tokenizer, device


def prompt(tokenizer, task: Task, text: str) -> str:
    # Clip the text itself: truncating the finished prompt would cut off the
    # assistant header at the end, which is the position we score.
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > MAX_TEXT_TOKENS:
        text = tokenizer.decode(ids[:MAX_TEXT_TOKENS])
    messages = [{"role": "user", "content": f"{task.instruction}\n\n{text}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def label_token_ids(tokenizer, task: Task) -> list[int]:
    ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in task.labels]
    assert len(set(ids)) == len(ids), f"labels of {task.name} share a first token"
    return ids


def examples(task: Task, split: str, n: int, seed: int = 0):
    data = load_dataset(task.dataset, split=split).shuffle(seed=seed)
    return data.select(range(min(n, len(data))))


@torch.no_grad()
def accuracy(model, tokenizer, device, task: Task, n: int = 500, batch_size: int = 16) -> float:
    data = examples(task, task.eval_split, n)
    label_ids = label_token_ids(tokenizer, task)
    model.eval()
    correct = 0
    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size]
        prompts = [prompt(tokenizer, task, text) for text in batch[task.text_field]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        last = model(**enc).logits[:, -1, :]
        pred = last[:, label_ids].argmax(dim=-1).cpu()
        correct += (pred == torch.tensor(batch["label"])).sum().item()
    return correct / len(data)

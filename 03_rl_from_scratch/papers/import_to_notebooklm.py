#!/usr/bin/env python3
"""Create RL-A and RL-B notebooks in Gemini Notebook (NotebookLM) via the official MCP endpoint."""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MCP_URL = "https://notebook.googleapis.com/mcp"
PAPERS_DIR = Path(__file__).resolve().parent


def get_token() -> str:
    return subprocess.check_output(
        ["gcloud", "auth", "application-default", "print-access-token"],
        text=True,
    ).strip()


def mcp_call(token: str, tool_name: str, arguments: dict, req_id: int = 1) -> dict:
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }).encode("utf-8")
    req = urllib.request.Request(
        MCP_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    result = data.get("result", {})
    if result.get("isError"):
        raise RuntimeError(f"MCP error in {tool_name}: {result}")
    text = result["content"][0]["text"]
    return json.loads(text)


NOTEBOOKS = [
    {
        "displayName": "RL-A：强化学习原理与工程驯服",
        "sources": [
            {
                "displayName": "A1_REINFORCE_Williams_1992",
                "webContent": {"url": "https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf"},
            },
            {
                "displayName": "A1b_Pong_from_Pixels_Karpathy_2016",
                "webContent": {"url": "https://karpathy.github.io/2016/05/31/rl/"},
            },
            {
                "displayName": "A2_GAE_Schulman_2015",
                "webContent": {"url": "https://arxiv.org/pdf/1506.02438.pdf"},
            },
            {
                "displayName": "A3_PPO_Schulman_2017",
                "webContent": {"url": "https://arxiv.org/pdf/1707.06347.pdf"},
            },
            {
                "displayName": "A4_PPO_37_Implementation_Details_2022",
                "webContent": {"url": "https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/"},
            },
            {
                "displayName": "A4b_Deep_RL_That_Matters_Henderson_2017",
                "webContent": {"url": "https://arxiv.org/pdf/1709.06560.pdf"},
            },
        ],
    },
    {
        "displayName": "RL-B：现代 LLM 强化学习实战",
        "sources": [
            {
                "displayName": "B1_InstructGPT_RLHF_Ouyang_2022",
                "webContent": {"url": "https://arxiv.org/pdf/2203.02155.pdf"},
            },
            {
                "displayName": "B2_DPO_Rafailov_2023",
                "webContent": {"url": "https://arxiv.org/pdf/2305.18290.pdf"},
            },
            {
                "displayName": "B3_DeepSeekMath_GRPO_2024",
                "webContent": {"url": "https://arxiv.org/pdf/2402.03300.pdf"},
            },
            {
                "displayName": "B3b_DeepSeek_R1_2025",
                "webContent": {"url": "https://arxiv.org/pdf/2501.12948.pdf"},
            },
            {
                "displayName": "B4_DAPO_ByteDance_2025",
                "webContent": {"url": "https://arxiv.org/pdf/2503.14476.pdf"},
            },
        ],
    },
]


def main() -> None:
    token = get_token()
    existing = mcp_call(token, "list_notebooks", {"pageSize": 50}).get("notebooks", [])
    by_name = {nb.get("displayName"): nb for nb in existing}

    created_notebooks = []
    req_id = 10
    for spec in NOTEBOOKS:
        title = spec["displayName"]
        if title in by_name:
            nb = by_name[title]
            print(f"[REUSE] {title} -> {nb['name']}")
        else:
            req_id += 1
            nb = mcp_call(token, "create_notebook", {"notebook": {"displayName": title}}, req_id)
            print(f"[CREATED] {title} -> {nb['name']}")
        nb_name = nb["name"]
        created_notebooks.append((title, nb_name))

        req_id += 1
        cur_sources = mcp_call(token, "list_sources", {"parent": nb_name, "pageSize": 50}, req_id).get("sources", [])
        existing_titles = {s.get("displayName") for s in cur_sources}

        for src in spec["sources"]:
            if src["displayName"] in existing_titles:
                print(f"  [SKIP] already exists: {src['displayName']}")
                continue
            req_id += 1
            res = mcp_call(token, "create_source", {"parent": nb_name, "source": src}, req_id)
            print(f"  [ADDED] {src['displayName']} -> state={res.get('state')} ({res.get('name')})")

    print("\nWaiting 10s for source ingestion...")
    time.sleep(10)
    for title, nb_name in created_notebooks:
        nb_id = nb_name.split("/")[-1]
        req_id += 1
        srcs = mcp_call(token, "list_sources", {"parent": nb_name, "pageSize": 50}, req_id).get("sources", [])
        print(f"\n=== {title} (https://notebooklm.google.com/notebook/{nb_id}) ===")
        for s in srcs:
            err = s.get("error")
            err_str = f" ERROR={err}" if err else ""
            print(f"  - {s.get('displayName')} | state={s.get('state')}{err_str}")


if __name__ == "__main__":
    main()

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph import graph
from state import ReflexionState

QUESTION = "What does LangGraph persistence do?"
PREVIEW = 80                                   # characters shown per value on screen

ALL_KEYS = list(ReflexionState.__annotations__.keys())

log_dir = Path(__file__).resolve().parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.jsonl"


def preview(value):
    text = str(value).replace("\n", " ")
    return text if len(text) <= PREVIEW else text[:PREVIEW] + "..."


def log(record):
    with open(log_file, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


step = 0
previous = {}
last_node = "(input)"

for mode, chunk in graph.stream(
    {"question": QUESTION}, stream_mode=["updates", "values"]
):
    if mode == "updates":
        # chunk looks like {"node_name": {...what the node returned...}}
        for node, update in chunk.items():
            last_node = node
            log({"step": step, "type": "update", "node": node, "update": update})

    elif mode == "values":
        changed = [k for k in chunk if k not in previous or chunk[k] != previous[k]]
        absent = [k for k in ALL_KEYS if k not in chunk]

        print(f"\n===== STEP {step}: state after {last_node} =====")
        for key, value in chunk.items():
            flag = "CHANGED  " if key in changed else "unchanged"
            print(f"  [{flag}] {key}: {preview(value)}")
        print(f"  [absent   ] {absent}")

        log({"step": step, "type": "state", "after": last_node,
             "changed": changed, "absent": absent, "state": chunk})
        previous = dict(chunk)
        step += 1

print(f"\nFull log written to: {log_file}")
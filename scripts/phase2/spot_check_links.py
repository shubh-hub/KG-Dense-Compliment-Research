#!/usr/bin/env python3
import json
import random
from pathlib import Path

def main():
    input_path = Path("data/phase2/annotated_12k_v5.jsonl")
    if not input_path.exists():
        print(f"ERROR: Input file not found at {input_path}")
        return

    print(f"Spot-checking 30 random linked mentions from {input_path}...\n")
    
    xs = [json.loads(ln) for ln in open(input_path, encoding="utf-8")]
    linked_mentions = []
    for x in xs:
        text = x.get("text") or x.get("question", "")
        for m in x.get("el_metadata", []):
            if m.get("qid") and m["qid"] != "NIL":
                linked_mentions.append((text, m["mention"], m["qid"], m.get("score", 0.0)))
    
    if not linked_mentions:
        print("No linked mentions found to sample.")
        return

    random.shuffle(linked_mentions)
    
    for text, mention, qid, score in linked_mentions[:30]:
        print(f"{mention} -> {qid} ({score:.2f}) | {text[:140]}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse, gzip, json
from pathlib import Path
from tqdm import tqdm

def open_any(p):
    return gzip.open(p, "rt", encoding="utf-8", errors="ignore") if str(p).endswith(".gz") else open(p, "r", encoding="utf-8", errors="ignore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity_labels", required=True,
                    help="Path to Wikidata5M entity-label file (e.g., wikidata5m_entity.txt or entity2text.txt) [QID\\tLabel]")
    ap.add_argument("--entity_desc", required=False,
                    help="Optional descriptions file (e.g., wikidata5m_text.txt.gz) [QID\\tDescription]")
    ap.add_argument("--out_json", required=True,
                    help="Output JSON (array) with fields: {entity_id, label, description, aliases: []}. Use '-' to skip array output.")
    ap.add_argument("--out_jsonl", required=False,
                    help="Optional JSONL streaming output file (one entity per line)")
    ap.add_argument("--flush_every", type=int, default=100000,
                    help="Flush progress every N records when streaming JSONL")
    args = ap.parse_args()

    print("[repair_entities_labels] Reading labels...")
    labels = {}
    with open_any(args.entity_labels) as f:
        for line in tqdm(f, desc="labels"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                qid, label = parts[0].strip(), parts[1].strip()
                if qid.startswith("Q") and label:
                    labels[qid] = label
    print(f"[repair_entities_labels] Loaded labels: {len(labels):,}")

    # Read descriptions (optional)
    desc = {}
    if args.entity_desc:
        print("[repair_entities_labels] Reading descriptions...")
        with open_any(args.entity_desc) as f:
            for line in tqdm(f, desc="descriptions"):
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    qid, d = parts[0].strip(), parts[1].strip()
                    if qid in labels:
                        desc[qid] = d
        print(f"[repair_entities_labels] Matched descriptions: {len(desc):,}")

    # Prepare outputs
    out_jsonl_path = args.out_jsonl
    jsonl_fh = None
    if out_jsonl_path:
        Path(out_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        jsonl_fh = open(out_jsonl_path, "w", encoding="utf-8")
        print(f"[repair_entities_labels] Streaming JSONL → {out_jsonl_path} (flush_every={args.flush_every})")

    out = [] if args.out_json != '-' else None
    print("[repair_entities_labels] Emitting entities...")
    count = 0
    for qid, label in tqdm(labels.items(), desc="emit"):
        obj = {"entity_id": qid, "label": label, "description": desc.get(qid, ""), "aliases": []}
        if jsonl_fh:
            jsonl_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
            if count % args.flush_every == 0:
                jsonl_fh.flush()
        if out is not None:
            out.append(obj)

    if jsonl_fh:
        jsonl_fh.flush()
        jsonl_fh.close()
        print(f"[repair_entities_labels] Finished JSONL streaming: {count:,} rows")

    if out is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[repair_entities_labels] Writing JSON array → {out_path}")
        with open(out_path, "w", encoding="utf-8") as fo:
            json.dump(out, fo, ensure_ascii=False, indent=2)
        print(f"[repair_entities_labels] Done: {len(out):,} records")

if __name__ == "__main__":
    main()

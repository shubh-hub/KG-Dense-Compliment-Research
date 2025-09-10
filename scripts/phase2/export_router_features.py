#!/usr/bin/env python3
import json
import csv
from pathlib import Path

def main():
    src_path = Path("data/phase2/annotated_12k_sliced_v5.jsonl")
    out_path = Path("data/phase2/router_dev_v5.csv")

    if not src_path.exists():
        print(f"ERROR: Source file not found at {src_path}")
        return

    with open(src_path, encoding="utf-8") as f_in, open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["slice", "n_mentions", "n_linked_qids", "n_conf_qids", "avg_qid_score", "nil_rate", "two_hop_reachable"])
        
        count = 0
        for ln in f_in:
            x = json.loads(ln)
            s = x.get("slice", "none")
            r = x.get("router_feats", {})
            writer.writerow([
                s, 
                r.get("n_mentions", 0), 
                r.get("n_linked_qids", 0), 
                r.get("n_conf_qids", 0),
                round(r.get("avg_qid_score", 0.0), 3), 
                round(r.get("nil_rate", 1.0), 3),
                int(bool(r.get("two_hop_reachable", False)))
            ])
            count += 1

    print(f"Successfully wrote {count} rows to {out_path}")

if __name__ == "__main__":
    main()

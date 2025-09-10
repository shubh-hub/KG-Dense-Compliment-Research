#!/usr/bin/env python3
import json, argparse
from pathlib import Path
from collections import Counter
import spacy

def norm(s): return " ".join(s.lower().split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--spacy_model", default="en_core_web_sm")
    args = ap.parse_args()

    print(f"[extract_mentions] Loading spaCy model: {args.spacy_model}")
    nlp = spacy.load(args.spacy_model, disable=["tagger","parser","lemmatizer","attribute_ruler"])
    
    print(f"[extract_mentions] Processing corpus: {args.corpus}")
    counts = Counter()
    processed = 0
    
    with open(args.corpus,"r",encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            text = x.get("text") or x.get("question") or ""
            if not text: continue
            
            for ent in nlp(text).ents:
                m = norm(ent.text)
                if m: counts[m] += 1
            
            processed += 1
            if processed % 1000 == 0:
                print(f"[extract_mentions] Processed {processed:,} docs, found {len(counts):,} unique mentions")

    print(f"[extract_mentions] Final: {processed:,} docs, {len(counts):,} unique mentions")
    
    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as fo:
        for m in counts.keys(): fo.write(m+"\n")
    with open(args.out_json, "w", encoding="utf-8") as fo:
        json.dump(counts.most_common(), fo, ensure_ascii=False, indent=2)
    
    print(f"[extract_mentions] Saved mentions to {args.out_txt} and {args.out_json}")

if __name__ == "__main__":
    main()

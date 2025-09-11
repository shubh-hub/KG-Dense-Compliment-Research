#!/usr/bin/env python3
import json, argparse, re, gzip
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from tqdm import tqdm

def gold_qid(s):
    return s.get("motif",[None]*5)[-1] if s.get("source")=="kg-qa-generated" else None

def load_labels(path="data/raw/kg/wikidata5m/wikidata5m_entity.txt"):
    d={}
    with open(path,encoding="utf-8") as f:
        for ln in f:
            p=ln.rstrip("\n").split("\t")
            if len(p)>=2: d[p[0]]=p[1]
    return d

def load_descriptions(path="data/raw/kg/wikidata5m/wikidata5m_text.txt.gz"):
    d = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for ln in f:
            parts = ln.strip().split('\t')
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--entity_labels", default="data/raw/kg/wikidata5m/wikidata5m_entity.txt")
    ap.add_argument("--entity_descriptions", default="data/raw/kg/wikidata5m/wikidata5m_text.txt.gz")
    ap.add_argument("--train_frac", type=float, default=0.7, help="Fraction of samples for training")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for train/test split")
    ap.add_argument("--model_name", default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    qid2label = load_labels(args.entity_labels)
    qid2desc = load_descriptions(args.entity_descriptions)

    samples = [json.loads(ln) for ln in open(args.in_jsonl, encoding="utf-8")]
    
    # Split samples
    np.random.seed(args.seed)
    gold_samples = [s for s in samples if gold_qid(s)]
    n_train = int(len(gold_samples) * args.train_frac)
    indices = np.random.permutation(len(gold_samples))
    train_samples = [gold_samples[i] for i in indices[:n_train]]
    test_samples = [gold_samples[i] for i in indices[n_train:]]
    print(f"Split: {len(train_samples)} train, {len(test_samples)} test samples")

    # 1. Prepare training data
    train_examples = []
    for s in tqdm(train_samples, desc="Preparing train data"):
        g = gold_qid(s)
        q = (s.get("text") or s.get("question",""))
        all_qids = {c['qid'] for c in s.get("text_candidates", [])} | {c['qid'] for c in s.get("kg_candidates", [])}
        for qid in all_qids:
            passage = qid2label.get(qid, "") + ". " + qid2desc.get(qid, "")
            label = 1 if qid == g else 0
            train_examples.append(InputExample(texts=[q, passage], label=float(label)))

    # 2. Train the CrossEncoder model
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = CrossEncoder(args.model_name, num_labels=1, max_length=512, device=device)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    
    print(f"Training on {device}...")
    model.fit(train_dataloader=train_dataloader,
              epochs=args.epochs,
              warmup_steps=100,
              output_path=f"models/{args.model_name.replace('/', '_')}-finetuned")

    # 3. Evaluate on test set
    print("Evaluating on test set...")
    hits = {1: 0, 5: 0, 10: 0, 20: 0}
    total = 0
    for s in tqdm(test_samples, desc="Evaluating"):
        g = gold_qid(s)
        if not g: continue
        total += 1
        q = (s.get("text") or s.get("question",""))
        all_qids = {c['qid'] for c in s.get("text_candidates", [])} | {c['qid'] for c in s.get("kg_candidates", [])}
        
        if not all_qids: continue

        passages = [(qid, qid2label.get(qid, "") + ". " + qid2desc.get(qid, "")) for qid in all_qids]
        scores = model.predict([[q, passage] for _, passage in passages])
        
        ranked_qids = [passages[i][0] for i in np.argsort(scores)[::-1]]
        
        for k in [1, 5, 10, 20]:
            if g in ranked_qids[:k]:
                hits[k] += 1

    test_recall = {k: hits[k] / max(1, total) for k in hits}
    print(f"Test Recall: @1={test_recall[1]:.3f}, @5={test_recall[5]:.3f}, @10={test_recall[10]:.3f}, @20={test_recall[20]:.3f}")

    # 4. Rerank and write output for all samples
    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        for s in tqdm(samples, desc="Reranking all samples"):
            q = (s.get("text") or s.get("question",""))
            all_qids = {c['qid'] for c in s.get("text_candidates", [])} | {c['qid'] for c in s.get("kg_candidates", [])}

            if all_qids:
                passages = [(qid, qid2label.get(qid, "") + ". " + qid2desc.get(qid, "")) for qid in all_qids]
                scores = model.predict([[q, passage] for _, passage in passages])
                ranked_qids = [passages[i][0] for i in np.argsort(scores)[::-1]]
                s["fused_candidates"] = [{"qid": qid, "score": float(scores[i])} for i, qid in enumerate(ranked_qids)][:20]

            out.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Wrote reranked output to {args.out_jsonl}")

if __name__=="__main__": main()

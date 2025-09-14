import json
import numpy as np
import argparse
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

def train_kge(triples_path, output_path, epochs=5, embedding_dim=100):
    """
    Trains a TransE model on the given KG triples and saves the artifacts.
    """
    print("Loading triples...")
    # Load triples from the JSONL file
    triples = []
    with open(triples_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                o = json.loads(line)
                triples.append([o['head'], o['relation'], o['tail']])
            except (json.JSONDecodeError, KeyError):
                continue

    if not triples:
        print("No triples found. Aborting.")
        return

    print(f"Loaded {len(triples)} triples.")
    
    # Create a TriplesFactory
    tf = TriplesFactory.from_labeled_triples(np.array(triples))

    # Run the PyKEEN pipeline
    print("Starting KGE model training...")
    result = pipeline(
        training=tf,
        model='TransE',
        model_kwargs=dict(embedding_dim=embedding_dim),
        training_kwargs=dict(num_epochs=epochs, use_tqdm_batch=False),
        random_seed=42,
        device='cpu', # Ensure it runs on CPU to avoid MPS/CUDA issues
    )
    print("Training complete.")

    model = result.model

    # Extract embeddings and mappings
    entity_embeddings = model.entity_representations[0](indices=None).detach().cpu().numpy()
    relation_embeddings = model.relation_representations[0](indices=None).detach().cpu().numpy()

    entity_to_id = tf.entity_to_id
    relation_to_id = tf.relation_to_id

    # Save the artifacts
    print(f"Saving model artifacts to {output_path}...")
    np.savez_compressed(
        output_path,
        e_map=entity_to_id,
        r_map=relation_to_id,
        e_emb=entity_embeddings,
        r_emb=relation_embeddings,
    )
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a TransE model on KG triples.')
    parser.add_argument('--triples_path', type=str, default='data/phase2/kg_subset_triples_expanded_v2.jsonl', help='Path to the KG triples JSONL file.')
    parser.add_argument('--output_path', type=str, default='models/kge_transe_100d.npz', help='Path to save the trained KGE model artifacts.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Dimension of the embeddings.')
    args = parser.parse_args()
    
    train_kge(args.triples_path, args.output_path, args.epochs, args.embedding_dim)

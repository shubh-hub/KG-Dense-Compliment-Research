from pykeen.pipeline import pipeline
import numpy as np
import json
import os

# Define paths
TRAIN_PATH = "data/phase3/kge/train.tsv"
OUTPUT_DIR = "data/phase3/kge/transE_v1"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Run the PyKEEN pipeline
res = pipeline(
    training=TRAIN_PATH,
    model='TransE',
    model_kwargs=dict(embedding_dim=200, scoring_fct_norm=1),
    training_kwargs=dict(num_epochs=150, batch_size=4096),
    stopper='early', stopper_kwargs=dict(patience=8),
    metadata=dict(title='transE_v1')
)

# Extract embeddings and mappings
E = res.model.entity_representations[0]().detach().cpu().numpy()
R = res.model.relation_representations[0]().detach().cpu().numpy()
eid = res.training.entity_to_id
rid = res.training.relation_to_id

# Save the artifacts as separate, portable files
np.save(os.path.join(OUTPUT_DIR, 'e_emb.npy'), E)
np.save(os.path.join(OUTPUT_DIR, 'r_emb.npy'), R)
with open(os.path.join(OUTPUT_DIR, 'e_map.json'), 'w') as f:
    json.dump(eid, f)
with open(os.path.join(OUTPUT_DIR, 'r_map.json'), 'w') as f:
    json.dump(rid, f)

print(f"Wrote KGE artifacts to directory: {OUTPUT_DIR}")

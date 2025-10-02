import os, json, gzip, sys, numpy as np, pandas as pd

def convert_artifacts():
    SRC = "data/phase3/kge/artifacts"
    DST = "data/phase3/kge/transE_v1"

    print(f"Starting artifact conversion...\nSRC: {SRC}\nDST: {DST}")

    entity_emb_path = os.path.join(SRC, "entity_embeddings.npy")
    relation_emb_path = os.path.join(SRC, "relation_embeddings.npy")
    ent_gz_path = os.path.join(SRC, "entity_to_id.tsv.gz")
    rel_gz_path = os.path.join(SRC, "relation_to_id.tsv.gz")

    required_files = [entity_emb_path, relation_emb_path, ent_gz_path, rel_gz_path]
    for f_path in required_files:
        if not os.path.exists(f_path):
            print(f"[FATAL] Required artifact file not found: {f_path}", file=sys.stderr)
            sys.exit(1)

    print("All required source artifact files found.")
    os.makedirs(DST, exist_ok=True)

    E = np.load(entity_emb_path)
    R = np.load(relation_emb_path)
    np.save(os.path.join(DST, "e_emb.npy"), E)
    np.save(os.path.join(DST, "r_emb.npy"), R)
    print("Embeddings copied and renamed.")

    def load_id_map_gz(path_gz):
        with gzip.open(path_gz, "rt") as f:
            # The gzipped TSV files from PyKEEN have a header row (e.g., 'id\tlabel')
            # Setting header=0 tells pandas to use the first row as column names.
            df = pd.read_csv(f, sep="\t", header=0)
        # The columns are now 'id' and 'label', but we need to handle both orders
        if 'id' not in df.columns or 'label' not in df.columns:
            # Fallback if header is missing or malformed
            df.columns = ['id', 'label']
        return {str(row["label"]): int(row["id"]) for _, row in df.iterrows()}

    e_map = load_id_map_gz(ent_gz_path)
    r_map_raw = load_id_map_gz(rel_gz_path)
    print("ID maps loaded from gzipped TSVs.")

    def short_rel(k):
        return k.split("/")[-1]
    r_map = {short_rel(k): v for k, v in r_map_raw.items()}
    print("Relation keys normalized.")

    json.dump(e_map, open(os.path.join(DST, "e_map.json"), "w"))
    json.dump(r_map, open(os.path.join(DST, "r_map.json"), "w"))
    print("JSON maps saved.")

    print("\n--- CONVERSION SUCCESSFUL ---")
    print(f"Wrote bundle to: {DST}")
    print(f"e_emb: {E.shape}, r_emb: {R.shape}, |entities| {len(e_map)}, |relations| {len(r_map)}")

if __name__ == '__main__':
    convert_artifacts()

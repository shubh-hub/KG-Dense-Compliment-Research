#!/usr/bin/env python3
"""
Fix corrupted entities.json by processing raw Wikidata5M files.
Extract entity data from wikidata5m_text.txt.gz and create clean JSON format.
"""

import gzip
import json
import os
import re
from pathlib import Path
from tqdm import tqdm

def extract_entities_from_raw():
    """Extract entity data from raw Wikidata5M text file."""
    
    # Paths
    raw_kg_dir = Path("data/raw/kg/wikidata5m")
    processed_kg_dir = Path("data/processed/kg")
    processed_kg_dir.mkdir(parents=True, exist_ok=True)
    
    text_file = raw_kg_dir / "wikidata5m_text.txt.gz"
    
    if not text_file.exists():
        print(f"Error: {text_file} not found")
        return False
    
    print("Processing raw Wikidata5M text file...")
    
    # Process text file to extract entity information
    entities_data = []
    
    print("Processing entity text data...")
    with gzip.open(text_file, 'rt', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing text")):
            try:
                line = line.strip()
                if not line:
                    continue
                
                # Parse line format: entity_id \t description
                parts = line.split('\t', 1)  # Split on first tab only
                if len(parts) >= 2:
                    entity_id = parts[0].strip()
                    description = parts[1].strip()
                    
                    # Extract label from description (first few words or sentence)
                    label = description.split('.')[0][:100] if description else entity_id
                    
                    # Include all entities from the text file
                    entity_data = {
                        "entity_id": entity_id,
                        "label": label,
                        "description": description,
                        "aliases": []  # Will be populated later if needed
                    }
                    entities_data.append(entity_data)
                
                # Process in batches to avoid memory issues - increase limit for real run
                if len(entities_data) >= 2000000:
                    break  # Process 2M entities for better coverage
                    
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Extracted {len(entities_data)} entities")
    
    # Save clean entities.json
    output_file = processed_kg_dir / "entities_clean.json"
    print(f"Saving clean entities to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(entities_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully created clean entities file with {len(entities_data)} entities")
    return True

def main():
    """Main function."""
    print("=== Fixing Corrupted Entities Data ===")
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent.parent)
    
    success = extract_entities_from_raw()
    
    if success:
        print("\n✅ Successfully fixed entities data!")
        print("Next: Update build_alias_map.py to use entities_clean.json")
    else:
        print("\n❌ Failed to fix entities data")

if __name__ == "__main__":
    main()

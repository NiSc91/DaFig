import pandas as pd
from bioc import biocxml
from collections import defaultdict
import json

def extract_document_info_v5(document):
    doc_id = document.id
    
    if not document.passages:
        print(f"Warning: Document {doc_id} has no passages")
        return None
    
    passage = document.passages[0]  # Only consider the first (and only) passage
    text = passage.text
    annotations = []
    relations = []
    attributes = {}
    
    # First, process all relations to extract attributes
    for relation in passage.relations:
        if relation.infons.get('type') == 'Attribute':
            attr_id = relation.id
            attr_type = relation.infons.get('attribute type')
            attributes[attr_id] = attr_type
    
    # Process annotations
    for annotation in passage.annotations:
        ann_type = annotation.infons.get('type')
        ann_attributes = []
        
        # Look for attributes that belong to this annotation
        for attr_id, attr_type in attributes.items():
            if attr_id.startswith(annotation.id[1:]):  # Assuming attribute IDs start with 'A' followed by annotation number
                ann_attributes.append({'type': attr_type, 'value': attr_type})  # Using attr_type as both type and value
        
        ann_info = {
            'id': annotation.id,
            'type': ann_type,
            'offset': annotation.locations[0].offset if annotation.locations else None,
            'length': annotation.locations[0].length if annotation.locations else None,
            'text': annotation.text,
            'attributes': ann_attributes
        }
        annotations.append(ann_info)
    
    # Process relations (excluding Attribute relations)
    for r in passage.relations:
        if r.infons.get('type') not in ['Attribute']:
            if len(r.nodes) >= 2:
                relations.append({
                    'id': r.id,
                    'type': r.infons.get('type'),
                    'arg1': r.nodes[0].refid,
                    'arg2': r.nodes[1].refid
                })
            else:
                print(f"Warning: Relation {r.id} in document {doc_id} does not have enough nodes")
    
    return {
        'doc_id': doc_id,
        'text': text,
        'annotations': annotations,
        'relations': relations
    }

def process_collection(bioc_file_path):
    documents_data = []
    
    with biocxml.iterparse(bioc_file_path) as reader:
        for document in reader:
            doc_info = extract_document_info_v5(document)
            if doc_info:
                documents_data.append(doc_info)
    
    return documents_data

# Process the collection
bioc_file_path = '../data/main_corpus.bioc'
processed_data = process_collection(bioc_file_path)

# Convert to DataFrame
df = pd.DataFrame(processed_data)

# Convert lists to JSON strings for storage in CSV
for col in ['annotations', 'relations']:
    df[col] = df[col].apply(json.dumps)

# Write to CSV
df.to_csv('processed_data.csv', index=False)

# If you want to read the CSV back into a DataFrame later:
# df_read = pd.read_csv('processed_data.csv')
# for col in ['annotations', 'relations', 'attributes']:
#     df_read[col] = df_read[col].apply(json.loads)
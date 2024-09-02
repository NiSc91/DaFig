import os
import sys
import pandas as pd
import json
from pybrat.parser import BratParser, Entity, Event, Example, Relation, Span

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = "brat_data/annotated"
sys.path.append(parent_dir)  # Add the parent folder to the module search path
sys.path.append(parent_dir+"/brat_peek")  # Add the parent folder to the module search path

# Collection handler
class CollectionHandler:
    def __init__(self, data_path):
        self.data_path = data_path

    def get_collections(self):
        collections = []
        for root, dirs, files in os.walk(self.data_path):
            for dir_name in dirs:
                collections.append(dir_name)
        return collections

    def get_collection_path(self, collection_name):
        collection_path = os.path.join(self.data_path, collection_name)
        if not os.path.exists(collection_path):
            raise ValueError("Collection '{}' does not exist".format(collection_name))
        return collection_path

    def get_subcollections(self, collection_name):
        collection_path = self.get_collection_path(collection_name)
        subcollections = []
        for root, dirs, files in os.walk(collection_path):
            for dir_name in dirs:
                subcollections.append(dir_name)
        return subcollections

    def extract_ids_and_collections(self, target_collection=None):
        ids_and_collections = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    collection = self.get_collection_from_path(file_path)
                    if collection and (collection == target_collection or target_collection is None):
                        file_id = os.path.splitext(file)[0]
                        ids_and_collections.append((file_id, collection))
        return ids_and_collections

    def get_collection_from_path(self, file_path):
        collection_path = os.path.dirname(file_path)
        collection = os.path.basename(collection_path)
        return collection

def process_brat_data(input_dir):
    brat = BratParser(error="ignore")
    examples = brat.parse(input_dir)
    
    # Helper function to extract attributes from brat annotation files
    def get_attr(doc_id):
        attributes = []
        ann_file = os.path.join(input_dir, f"{doc_id}.ann")
        with open(ann_file, "r") as f:
            for line in f:
                if line.startswith("A"):
                    attr_id, info = line.strip().split("\t")
                    attr_type, ref_id, attr_value = info.split()
                    attr_info = {
                        "id": attr_id,
                        "type": attr_type,
                        "ref_id": ref_id,
                        "value": attr_value
                        }
                    attributes.append(attr_info)
        return attributes

    data = []
    for example in examples:
        doc_id = example.id
        text = example.text
        annotations = []
        relations = []
        attributes = []

        for entity in example.entities:
            # Get annotation info
            for span in entity.spans:
                entity_text = text[span.start:span.end]
                ann_info = {
                    'id': entity.id,
                    'type': entity.type,
                    'start': span.start,
                    'end': span.end,
                    'text': entity_text
                }
                annotations.append(ann_info)

        # Get relations
        for relation in example.relations:
            rel_info = {
                'id': relation.id,
                'type': relation.type,
                'arg1': relation.arg1,
                'arg2': relation.arg2
            }
            relations.append(rel_info)

        # Get attributes
        attributes = get_attr(doc_id)

        data.append({
            'doc_id': doc_id,
            'text': text,
            'annotations': annotations,
            'relations': relations,
            'attributes': attributes
        })

    # Transform lists to json format inside the dataframe
    df = pd.json_normalize(data)

    return df

# Usage
input_dir = os.path.join(CollectionHandler(data_path).get_collection_path("main_collection"), "combined_main")
output_file = "processed_brat_data.csv"

# Process the data
df = process_brat_data(input_dir)

# Save to CSV
df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")

## Create a new column, span, in the data-frame, infered from the start and end columns.

## Group MWEs based on their spans
from dataclasses import dataclass

@dataclass
class Span:
    start: int
    end: int

def group_mwe_relations(relations):
    mwes = []
    current_mwe = []
    
    for relation in relations:
        if relation['type'] == 'MultiWordExpression':
            arg1, arg2 = relation['arg1'], relation['arg2']
            
            if not current_mwe or arg1 in current_mwe:
                if arg1 not in current_mwe:
                    current_mwe.append(arg1)
                if arg2 not in current_mwe:
                    current_mwe.append(arg2)
            else:
                if current_mwe:
                    mwes.append(create_mwe_dict(current_mwe))
                current_mwe = [arg1, arg2]

    if current_mwe:
        mwes.append(create_mwe_dict(current_mwe))

    return mwes

def create_mwe_dict(entities):
    # Remove duplicate spans
    unique_spans = []
    seen = set()
    for entity in entities:
        span = Span(entity.spans[0].start, entity.spans[0].end)
        if (span.start, span.end) not in seen:
            unique_spans.append(span)
            seen.add((span.start, span.end))
    
    unique_spans.sort(key=lambda x: x.start)
    
    mwe_type = 'ContiguousMWE'
    if any(unique_spans[i+1].start - unique_spans[i].end > 1 for i in range(len(unique_spans)-1)):
        mwe_type = 'Non-contiguousMWE'
    
    # Remove duplicate words in mention
    unique_words = []
    for entity in entities:
        if entity.mention not in unique_words:
            unique_words.append(entity.mention)
    mention = ' '.join(unique_words)
    
    entity_type = entities[0].type  # Assuming all entities in an MWE have the same type
    
    # Remove duplicate ref_ids
    unique_ref_ids = list(dict.fromkeys(entity.id for entity in entities))
    
    return {
        'type': mwe_type,
        'mention': mention,
        'entity_type': entity_type,
        'spans': unique_spans,
        'ref_ids': unique_ref_ids
    }

# Apply the function to your data
df['grouped_mwes'] = df['relations'].apply(group_mwe_relations)

# Example usage
print(df.iloc[93]['grouped_mwes'])

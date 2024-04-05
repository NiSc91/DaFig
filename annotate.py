# Convert an ann-file into json format

import json
import os

def get_file_ids(directory):
    file_ids = []
    for filename in os.listdir(directory):
        if filename.endswith('.ann'):
            file_id = filename.split('.')[0]
            file_ids.append(file_id)
    return file_ids

def parse_ann_data(file_id, directory):
    with open(os.path.join(directory, f'{file_id}.ann'), 'r', encoding='utf-8') as f:
        data = f.read()

    annotations = {}
    for line in data.split('\n'):
        if line:
            parts = line.split('\t')
            if parts[0].startswith('T'):
                details = parts[1].split()
                annotations[parts[0]] = {'ID': id, 'annotation_id': parts[0], 'type': details[0], 'offsets': (details[1], details[2]), 'text': parts[2], 'attributes': [], 'relations': []}
            elif parts[0].startswith('A'):
                details = parts[1].split()
                if details[1] in annotations:
                    annotations[details[1]]['attributes'].append({'id': parts[0], 'type': details[0], 'value': parts[2]})
            elif parts[0].startswith('R'):
                details = parts[1].split()
                if details[1].split(':')[1] in annotations:
                    annotations[details[1].split(':')[1]]['relations'].append({'id': parts[0], 'type': details[0], 'arg2': details[2].split(':')[1]})
    return list(annotations.values())

def write_annotations_to_json(directory):
    """
    The `write_annotations_to_json` function collects all the parsed annotation data and writes it to a JSON file. Each annotation is a dictionary with the following keys:
    
    - "ID": The unique ID derived from the filename
    - "annotation_id": The ID of the annotation (e.g., 'T6', 'T7', etc.)
    - "type": The type of the annotation (e.g., 'HPB', 'MTP', etc.)
    - "offsets": A tuple representing the character offsets within the text
    - "text": The actual text that was annotated
    - "attributes": A list of dictionaries, where each dictionary represents an attribute associated with the annotation (with 'id', 'type', and 'value' keys)
    - "relations": A list of dictionaries, where each dictionary represents a relation in which the annotation is involved (with 'id', 'type', and 'arg2' keys)

The JSON file contains a list of such dictionaries - one for each annotation in all the .ann files in the directory. The structure of the data in the file makes it easily readable and suitable for further processing or analysis."""
    
    file_ids = get_file_ids(directory)
    all_annotations = []
    for file_id in file_ids:
        parsed_data = parse_ann_data(file_id, directory)
        all_annotations.extend(parsed_data)
    with open(f'{directory}_annotations.json', 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, indent=4)

def load_annotations(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_annotations(id, annotations1, annotations2):
    annotation1 = next((a for a in annotations1 if a['ID'] == id), None)
    annotation2 = next((a for a in annotations2 if a['ID'] == id), None)
    # TODO: Implement comparison logic here    
return

#directory = '/path/to/your/files'  # replace with your directory
#file_ids = get_file_ids(directory)
#file_id = '9450572'  # replace this with your unique ID
#parsed_data = parse_ann_data(unique_id)
#annotations1 = load_annotations('annotations_annotator-1.json')
#annotations2 = load_annotations('annotations_annotator-2.json')

""" fTODO for calculation of inter-annotator agreement (Cohen's kappa).

1. **Prepare the data:**
    - Extract the relevant data from your annotation data, i.e., the annotated items, their types, and attributes.
    - Ensure that the data is in a format suitable for the Cohen's Kappa calculation, typically a confusion matrix or a similar structure.

2. **Calculate Cohen's Kappa for annotated items:**
    - For each item (text span) in your documents, determine whether both annotators identified it as significant.
    - Calculate Cohen's Kappa to quantify the level of agreement.

3. **Calculate Cohen's Kappa for annotation types:**
    - For each annotated item, determine the type assigned by each annotator.
    - Calculate Cohen's Kappa for these type assignments to quantify the level of agreement.

4. **Calculate Cohen's Kappa for annotation attributes:**
    - For each annotated item, determine the attributes assigned by each annotator.
    - Calculate Cohen's Kappa for these attribute assignments to quantify the level of agreement.

5. **Interpret the results:**
    - Analyze the Cohen's Kappa values to understand where your annotators tend to agree and where disagreements are more common.
    - Use this analysis to identify areas where your annotation guidelines may need to be clarified or expanded."""
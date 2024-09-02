from bioc import biocxml
import pandas as pd
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
import torch_directml
# Load spaCy model for Danish
nlp = spacy.load("da_core_news_sm")

# Set up DirectML device
dml = torch_directml.device()

def process_document(document, tokenizer, nlp, max_length=512, print_sentences=10):
    print(f"Processing document: {document.id}")
    text = document.passages[0].text
    annotations = document.passages[0].annotations
    print(f"Document length: {len(text)} characters")
    print(f"Number of annotations: {len(annotations)}")

    doc = nlp(text)
    sentences = list(doc.sents)
    print(f"Total number of sentences: {len(sentences)}")

    result = []
    current_offset = 0

    for i, sentence in enumerate(sentences):
        encoding = tokenizer(sentence.text, return_offsets_mapping=True, add_special_tokens=True, 
                             max_length=max_length, truncation=True, padding='max_length')
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        offset_mapping = encoding['offset_mapping']

        metaphor_labels = [0] * len(input_ids)
        irony_labels = [0] * len(input_ids)
        hyperbole_labels = [0] * len(input_ids)

        offset_mapping = [(o[0] + current_offset, o[1] + current_offset) if o[0] != o[1] else (0, 0) for o in offset_mapping]

        for ann in annotations:
            start, end = ann.locations[0].offset, ann.locations[0].offset + ann.locations[0].length
            for j, (token_start, token_end) in enumerate(offset_mapping):
                if token_start >= start and token_end <= end:
                    if ann.infons['type'] == 'MTP':
                        metaphor_labels[j] = 1
                    elif ann.infons['type'] == 'VIR':
                        irony_labels[j] = 1
                    elif ann.infons['type'] == 'HPB':
                        hyperbole_labels[j] = 1

        # Print detailed information for the first 'print_sentences' sentences
        if i < print_sentences:
            print(f"\nProcessing sentence {i+1}: {sentence.text}")
            print("\nDetailed breakdown:")
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            print(f"{'Token':<15} {'Metaphor':<10} {'Irony':<10} {'Hyperbole':<10}")
            print("-" * 50)
            for token, met, iro, hyp in zip(tokens, metaphor_labels, irony_labels, hyperbole_labels):
                if token not in ('[PAD]', '[CLS]', '[SEP]'):  # Skip special tokens for clarity
                    print(f"{token:<15} {met:<10} {iro:<10} {hyp:<10}")

        result.append({
            'input_ids': torch.tensor(input_ids, device=dml),
            'attention_mask': torch.tensor(attention_mask, device=dml),
            'metaphor_labels': torch.tensor(metaphor_labels, device=dml),
            'irony_labels': torch.tensor(irony_labels, device=dml),
            'hyperbole_labels': torch.tensor(hyperbole_labels, device=dml)
        })

        current_offset += len(sentence.text)

    print(f"\nProcessed {len(result)} chunks for this document")
    print(result[:10])
    return result

## Debug version
def process_document_debug(document, tokenizer, nlp, max_length=512, print_sentences=10):
    print(f"Processing document: {document.id}")
    text = document.passages[0].text
    annotations = document.passages[0].annotations
    print(f"Document length: {len(text)} characters")
    print(f"Number of annotations: {len(annotations)}")
    
    # Print out all annotations
    print("\nAnnotations:")
    for ann in annotations:
        start, end = ann.locations[0].offset, ann.locations[0].offset + ann.locations[0].length
        print(f"Type: {ann.infons['type']}, Span: '{text[start:end]}', Offsets: {start}-{end}")

    doc = nlp(text)
    sentences = list(doc.sents)
    print(f"Total number of sentences: {len(sentences)}")

    result = []
    current_offset = 0

    for i, sentence in enumerate(sentences):
        encoding = tokenizer(sentence.text, return_offsets_mapping=True, add_special_tokens=True, 
                             max_length=max_length, truncation=True, padding='max_length')
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        offset_mapping = encoding['offset_mapping']

        metaphor_labels = [0] * len(input_ids)
        irony_labels = [0] * len(input_ids)
        hyperbole_labels = [0] * len(input_ids)

        offset_mapping = [(o[0] + current_offset, o[1] + current_offset) if o[0] != o[1] else (0, 0) for o in offset_mapping]

        label_counts = {'MTP': 0, 'VIR': 0, 'HPB': 0}

        for ann in annotations:
            start, end = ann.locations[0].offset, ann.locations[0].offset + ann.locations[0].length
            for j, (token_start, token_end) in enumerate(offset_mapping):
                if token_start >= start and token_end <= end:
                    if ann.infons['type'] == 'MTP':
                        metaphor_labels[j] = 1
                        label_counts['MTP'] += 1
                    elif ann.infons['type'] == 'VIR':
                        irony_labels[j] = 1
                        label_counts['VIR'] += 1
                    elif ann.infons['type'] == 'HPB':
                        hyperbole_labels[j] = 1
                        label_counts['HPB'] += 1

        # Print detailed information for the first 'print_sentences' sentences
        if i < print_sentences:
            print(f"\nProcessing sentence {i+1}: {sentence.text}")
            print(f"Sentence offset: {current_offset}-{current_offset + len(sentence.text)}")
            print(f"Label counts: Metaphor: {label_counts['MTP']}, Irony: {label_counts['VIR']}, Hyperbole: {label_counts['HPB']}")
            print("\nDetailed breakdown:")
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            print(f"{'Token':<15} {'Offset':<15} {'Metaphor':<10} {'Irony':<10} {'Hyperbole':<10}")
            print("-" * 65)
            for token, (start, end), met, iro, hyp in zip(tokens, offset_mapping, metaphor_labels, irony_labels, hyperbole_labels):
                if token not in ('[PAD]', '[CLS]', '[SEP]'):  # Skip special tokens for clarity
                    print(f"{token:<15} {f'{start}-{end}':<15} {met:<10} {iro:<10} {hyp:<10}")

        result.append({
            'input_ids':torch.tensor(input_ids, device=dml),
            'attention_mask': torch.tensor(attention_mask, device=dml),
            'metaphor_labels': torch.tensor(metaphor_labels, device=dml),
            'irony_labels': torch.tensor(irony_labels, device=dml),
            'hyperbole_labels': torch.tensor(hyperbole_labels, device=dml)
        })

        current_offset += len(sentence.text)

    print(f"\nProcessed {len(result)} chunks for this document")
    return result

# Load the tokenizer and model for the Danish BERT
tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")
model = AutoModel.from_pretrained("Maltehb/danish-bert-botxo").to(dml)

# Define file path to the BioC XML file
bioc_file_path = '../data/main_corpus.bioc'

# Deserialize ``fp`` to a BioC collection object.
with open(bioc_file_path, 'r') as fp:
    collection = biocxml.load(fp)

# Process all documents
for i, doc in enumerate(collection.documents):
    print(f"\n--- Processing document {i+1} ---")
    processed_data = process_document_debug(doc, tokenizer, nlp)
    processed_data.extend(processed_data)
    print(f"Total processed chunks so far: {len(processed_data)}")

# Example of using the model with DirectML
def process_batch(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    return outputs.last_hidden_state

# Process a batch
batch_size = 8
for i in range(0, len(processed_data), batch_size):
    batch = processed_data[i:i+batch_size]
    outputs = process_batch(batch)
    # Further processing with outputs...
    pass
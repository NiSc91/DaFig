from lxml import etree

def get_tag_examples(element, level=1, tag_dict=None):
    if tag_dict is None:
        tag_dict = {}
    
    tag = element.tag
    if tag not in tag_dict:
        tag_dict[tag] = (level, element)
    
    for child in element:
        get_tag_examples(child, level + 1, tag_dict)
    
    return tag_dict

# Define file path to the BioC XML file
bioc_file_path = 'main_corpus.bioc'

# Parse the XML file
tree = etree.parse(bioc_file_path)
root = tree.getroot()

# Get tag examples
tag_examples = get_tag_examples(root)

# Sort tags by level
sorted_tags = sorted(tag_examples.items(), key=lambda x: x[1][0])

# Print the results
for tag, (level, element) in sorted_tags:
    print(f"Level {level}: <{tag}>")
    
    # Print attributes if any
    if element.attrib:
        print("  Attributes:")
        for key, value in element.attrib.items():
            print(f"    {key}: {value}")
    
    # Print text content if any (truncated if too long)
    if element.text and element.text.strip():
        text = element.text.strip()
        if len(text) > 50:
            text = text[:47] + "..."
        print(f"  Text: {text}")
    
    print()  # Empty line for readability

import xml.etree.ElementTree as ET
import csv
import re

def danish_sentence_tokenize(text):
    return re.split(r'(?<=[.!?\n])', text)

def process_xml_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    results = []

    for document in root.findall('.//document'):
        doc_id = document.find('id').text
        
        for passage in document.findall('passage'):
            text = passage.find('text').text
            sentences = danish_sentence_tokenize(text)
            
            annotations = {}
            for annotation in passage.findall('annotation'):
                ann_id = annotation.get('id')
                ann_text = annotation.find('text').text
                offset = int(annotation.find('location').get('offset'))
                length = int(annotation.find('location').get('length'))
                annotations[ann_id] = {
                    'text': ann_text,
                    'offset': offset,
                    'length': length
                }

            relations = {}
            for relation in passage.findall('relation'):
                rel_id = relation.get('id')
                if rel_id.startswith('R'):
                    relation_type = relation.find('infon[@key="relation type"]').text
                    arg1 = arg2 = None
                    for node in relation.findall('node'):
                        role = node.get('role')
                        if role == 'Arg1':
                            arg1 = node.get('refid')
                        elif role == 'Arg2':
                            arg2 = node.get('refid')
                    if arg1 and arg2:
                        relations[rel_id] = {'type': relation_type, 'arg1': arg1, 'arg2': arg2}

            # Sort relations by their offset
            sorted_relations = sorted(relations.items(), key=lambda x: annotations[x[1]['arg1']]['offset'])

            combined_expressions = {}
            current_mwe = None
            for rel_id, rel in sorted_relations:
                if rel['arg1'] in annotations and rel['arg2'] in annotations:
                    arg1 = annotations[rel['arg1']]
                    arg2 = annotations[rel['arg2']]
                    
                    if current_mwe and arg1['offset'] == current_mwe['end_offset']:
                        # Extend the current MWE
                        current_mwe['text'] += f" {arg2['text']}"
                        current_mwe['end_offset'] = arg2['offset'] + arg2['length']
                        current_mwe['length'] = current_mwe['end_offset'] - current_mwe['offset']
                    else:
                        # Start a new MWE
                        if current_mwe:
                            combined_expressions[f"MWE_{len(combined_expressions)}"] = current_mwe
                        current_mwe = {
                            'text': f"{arg1['text']} {arg2['text']}",
                            'offset': arg1['offset'],
                            'end_offset': arg2['offset'] + arg2['length'],
                            'length': arg2['offset'] + arg2['length'] - arg1['offset'],
                            'type': 'MultiWordExpression'
                        }

            if current_mwe:
                combined_expressions[f"MWE_{len(combined_expressions)}"] = current_mwe

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_start = text.index(sentence)
                sentence_end = sentence_start + len(sentence)
                
                mtp_in_sentence = []
                for expr_id, expr in combined_expressions.items():
                    if sentence_start <= expr['offset'] < sentence_end:
                        mtp_in_sentence.append((expr['text'], expr['type']))
                
                for ann_id, ann in annotations.items():
                    if sentence_start <= ann['offset'] < sentence_end and ann['text'] not in ' '.join([mtp[0] for mtp in mtp_in_sentence]):
                        mtp_in_sentence.append((ann['text'], 'single_word'))
                
                if mtp_in_sentence:
                    results.append({
                        'doc_id': doc_id,
                        'sentence': sentence,
                        'mtp_words': ', '.join([f"{mtp[0]}|{mtp[1]}" for mtp in mtp_in_sentence])
                    })

    return results

def save_to_csv(results, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['doc_id', 'sentence', 'mtp_words', 'mtp_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            mtp_words = row['mtp_words'].split(', ')
            for mtp in mtp_words:
                word, mtp_type = mtp.split('|')
                writer.writerow({
                    'doc_id': row['doc_id'],
                    'sentence': row['sentence'],
                    'mtp_words': word,
                    'mtp_type': mtp_type
                })

results = process_xml_file(bioc_file_path)
save_to_csv(results, 'metaphors_main_corpus.csv')

import csv
from collections import defaultdict

# Initialize a dictionary to store metaphor words and their sentences
metaphor_dict = defaultdict(set)

# Initialize variables for statistics
total_tokens = 0
all_sentences = []
all_metaphor_tokens = []

# Read the CSV file
with open('metaphors_main_corpus.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        sentence = row['sentence']
        all_sentences.append(sentence)
        mtp_word = row['mtp_words']
        mtp_type = row['mtp_type']
        all_metaphor_tokens.append(mtp_word)

        metaphor_dict[mtp_word].add((sentence, mtp_type))

# Calculate statistics
total_tokens = sum(len(sentence.split()) for sentence in all_sentences)
metaphor_type_count = len(metaphor_dict)
metaphor_token_count = len(all_metaphor_tokens)

metaphor_proportion = (metaphor_token_count / total_tokens) * 100

# Calculate statistics for each metaphor type
single_word_count = sum(1 for word, sentences in metaphor_dict.items() if any(t == 'single_word' for _, t in sentences))
mwe_count = sum(1 for word, sentences in metaphor_dict.items() if any(t == 'MultiWordExpression' for _, t in sentences))
chunk_count = sum(1 for word, sentences in metaphor_dict.items() if any(t == 'CHUNK' for _, t in sentences))

print(f"Total tokens: {total_tokens}")
print(f"Metaphor types: {metaphor_type_count}")
print(f"Metaphor tokens: {metaphor_token_count}")
print(f"Metaphor proportion: {metaphor_proportion:.2f}%")
print(f"Single word metaphors: {single_word_count}")
print(f"Multi-word expressions: {mwe_count}")
print(f"Chunks: {chunk_count}")

# Write the results to a new file
with open('metaphor_summary.txt', 'w', encoding='utf-8') as outfile:
    # Write statistics
    outfile.write("Statistics:\n")
    outfile.write(f"Number of unique metaphor words (types): {metaphor_type_count}\n")
    outfile.write(f"Total words (tokens) in corpus: {total_tokens}\n")
    outfile.write(f"Proportion of metaphor tokens: {metaphor_proportion:.2f}%\n\n")
    outfile.write("=" * 50 + "\n\n")

    # Write metaphor words and example sentences
    for word, sentences in sorted(metaphor_dict.items()):
        outfile.write(f"Metaphor word: {word}\n")
        outfile.write("Example sentences:\n")
        for sentence in sentences:
            outfile.write(f"- {sentence}\n")
        outfile.write("\n")

print("Metaphor summary with statistics has been written to metaphor_summary.txt")

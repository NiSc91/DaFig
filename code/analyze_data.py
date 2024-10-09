import os
import shutil
from config import *
#from nltk import RegexpTokenizer
import re
import peek
import pandas as pd
import numpy as np
import json
from config import *
from parse_data import ExtendedExample, ExtendedBratParser, LexicalUnit, Span, Entity, Relation, Attribute
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from itertools import chain
from dataclasses import dataclass
from matplotlib import pyplot as plt

## Declare variables
handler = CollectionHandler(CORPORA_DIR)

all_corpora = handler.get_collections()
print("All annotated corpora:", all_corpora)
MAIN_PATH = handler.get_collection_path(os.path.join(CORPORA_DIR, 'main'))
AGR1_PATH = handler.get_collection_path(os.path.join(CORPORA_DIR, 'agr1'))
AGR2_PATH = handler.get_collection_path(os.path.join(CORPORA_DIR, 'agr2'))
AGR3_PATH = handler.get_collection_path(os.path.join(CORPORA_DIR, 'agr3'))
AGR_FINAL_PATH = handler.get_collection_path(os.path.join(CORPORA_DIR, 'agr_final'))
CONSENSUS_PATH = handler.get_collection_path(os.path.join(CORPORA_DIR, 'consensus_agr2'))
OUTPUT_DIR = TEMP_DIR

## Read in the main corpus and perform some basic operations
main_corpus = peek.AnnCorpus(MAIN_PATH, txt=True)
doc = main_corpus.get_random_doc()
print("Random corpus document from the main collection:")
print()
print(doc.anns.items())

### Rudimentary analysis using brat_peek library ###

def analyze_corpus(input_dir, output_dir):
    # Initialize corpus
    corpus = peek.AnnCorpus(input_dir, txt=True)

    # Generate basic statistics
    print('Corpus stats:', corpus.count)
    print('Entity labels found in corpus:', corpus.text_labels)
    print('Document 42 annotations:', corpus.docs[41].anns)

    # Generate and save plot
    peek.stats.plot_tags(corpus, save_fig=True, outpath=os.path.join(output_dir, 'corpus_stats.png'))

    # Generate .tsv with statistics
    peek.stats.generate_corpus_stats_tsv(corpus, include_txt=True, out_path=output_dir)

    # Filter text frequency to exclude stop words
    stop_words = set(stopwords.words('danish'))
    filtered_text_freq = {
        entity: Counter({word: freq for word, freq in text_freq.items() if word not in stop_words})
        for entity, text_freq in corpus.text_freq.items()
    }

    # Get the 10 most common metaphors and hyperboles
    print("10 most common metaphors:", filtered_text_freq['MTP'].most_common(10))
    print("10 most common hyperboles:", filtered_text_freq['HPB'].most_common(10))

    return corpus, filtered_text_freq

### Parse DaFig data ###

brat = ExtendedBratParser(input_dir=MAIN_PATH, error="ignore")
examples = list(brat.parse(MAIN_PATH))

# Find misaligned spans, i.e. instances where an entity span/mention does not correspond to a token in the text

def find_misaligned_spans(examples):
    misaligned_spans = defaultdict(list)
    for example in examples:
        doc_id = example.id; text = example.text; entities = example.entities
        for entity in entities:
            is_misaligned = False
            # Check if the character in the text before or after a span is alphanumeric
            if entity.start > 0 and text[entity.start - 1].isalnum():
                is_misaligned = True
            elif entity.end < len(text) and text[entity.end].isalnum():
                is_misaligned = True
            
            if is_misaligned:
                misaligned_spans[doc_id].append((entity))

    return misaligned_spans

# Semi-automatic approach to suggest full word annotations
def suggest_full_word_annotation(text, start, end):
    # Find word boundaries
    word_start = start
    while word_start > 0 and text[word_start-1].isalnum():
        word_start -= 1
    
    word_end = end
    while word_end < len(text) and text[word_end].isalnum():
        word_end += 1
    
    return word_start, word_end


def write_misaligned_spans(corpus_path, output_dir):
    # Extract the corpus name from the path
    corpus_name = os.path.basename(corpus_path)
    
    # Initialize the ExtendedBratParser
    brat = ExtendedBratParser(input_dir=corpus_path, error="ignore")
    
    # Parse the corpus
    examples = list(brat.parse(corpus_path))
    
    # Find misaligned spans
    misaligned = find_misaligned_spans(examples)
    
    # Count the number of misaligned entities in the corpus
    total_misaligned = sum(len(misaligned[doc_id]) for doc_id in misaligned)
    print(f"Total misaligned entities: {total_misaligned}")
    
    # Construct the output filename
    output_filename = f"misaligned_spans_{corpus_name}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # Write the misaligned spans and the full word annotations to a json file
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            if example.id in misaligned:
                for entity in misaligned[example.id]:
                    start, end = suggest_full_word_annotation(example.text, entity.start, entity.end)
                    suggested_mention = example.text[start:end]
                    
                    data = {
                        "doc_ID": example.id,
                        'entity_ID': entity.id,
                        "original_mention": entity.mention,
                        "original_span": {"start": entity.spans[0].start, "end": entity.spans[0].end},
                        "suggested_mention": suggested_mention,
                        "suggested_span": {"start": start, "end": end}
                    }
                    
                    json.dump(data, f, ensure_ascii=False)
                    f.write('\n')
    
    print(f"Results written to: {output_path}")

# Do this for all corpora
misaligned_main = write_misaligned_spans(MAIN_PATH, OUTPUT_DIR)
misaligned_agr1 = write_misaligned_spans(AGR1_PATH, OUTPUT_DIR)
misaligned_agr2 = write_misaligned_spans(AGR2_PATH, OUTPUT_DIR)
misaligned_agr3 = write_misaligned_spans(AGR3_PATH, OUTPUT_DIR)
misaligned_agr_final = write_misaligned_spans(AGR_FINAL_PATH, OUTPUT_DIR)
misaligned_consensus_agr2 = write_misaligned_spans(CONSENSUS_PATH, OUTPUT_DIR)

### Gather advanced stats ###

# Get lexical units info:
lexical_units = [lu for example in examples for lu in example.lexical_units]
lu_counts = len(lexical_units)
# Get the number of tokens for all lexical units
token_counts = sum(len(lu.mention.split()) for lu in lexical_units)

# Get tag info
lu_tag_counts = Counter(lu.tag for lu in lexical_units)

# Get the proportion of each tag in the lexical units
tag_proportions = {tag: count / lu_counts for tag, count in lu_tag_counts.items()}

# Get MWE and chunk info:
mwe_chunk_counts = sum(len(lu.spans) > 1 for lu in lexical_units)
lu_type_counts = Counter(lu.type for lu in lexical_units)

### Frequency information ###

# Get the 10 most common lexical units tagged with MTP (use counter to get the most common elements)
mtp_lus = [lu for lu in lexical_units if lu.tag == 'MTP']
mtp_lu_counts = Counter(lu.mention for lu in mtp_lus)
most_common_mtp_lus = mtp_lu_counts.most_common(10)

### Output results ###

print(f"Total lexical units: {lu_counts}")
print(f"Tag counts: {lu_tag_counts}")
print(f"Type counts: {lu_type_counts}")
print(f"10 most common MTP lexical units: {most_common_mtp_lus}")

## Count absolute amount of tokens in the corpus
tokens = sum(len(example.text.split()) for example in examples)
print(f"Total tokens in the corpus: {tokens}")

### Create dictionaries to better understand the data ###

examples = list(brat.parse(MAIN_PATH))

# Create a dictionary with doc_id as key, and the text as value.
all_texts = {
    example.id: example.text
    for example in examples}

## Create an all_lexical_units dictionary with document_ID as key, and a list of lexical units as values.
all_lus = {
    example.id: [(lu.mention, lu.spans, lu.type, lu.tag) for lu in example.lexical_units]
    for example in examples}

# Based on the all_lus dictionary, create a new dictionary with document_ID as key, and a list of entities which have different tags but same or over-lapping spans, as values.
lus_with_different_types = defaultdict(list)

for doc_id, lus in all_lus.items():
    for lu1, spans1, type1, tag1 in lus:
        for lu2, spans2, type2, tag2 in lus:
            if tag1!= tag2 and spans1 == spans2:
                lus_with_different_types[doc_id].append((lu1, type1, tag1, lu2, type2, tag2))

# Avoid duplicates by converting the list to a set
lus_with_different_types = {doc_id: list(set(lus)) for doc_id, lus in lus_with_different_types.items()}

# Output the results to a txt-file and save to OUTPUT_DIR as overlaps.txt
with open(os.path.join(OUTPUT_DIR, "lu_overlaps.txt"), "w") as f:
    for doc_id, lus in lus_with_different_types.items():
        f.write(f"Document ID: {doc_id}\n")
        f.write(f"Annotations with different tags but same spans:\n")

        for lu1, type1, tag1, lu2, type2, tag2 in lus:
            f.write(f"Entity 1: {lu1} ({type1} - {tag1})\n")
            f.write(f"Entity 2: {lu2} ({type2} - {tag2})\n")
            f.write("\n")

### Visualizations ###

# Plot the distribution of lexical unit types and figures of speech in the corpus combined into a single plot

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Subplot 1: Distribution of Tags (Figures of Speech)
tag_mapping = {'MTP': 'Metaphor', 'VIR': 'Irony', 'HPB': 'Hyperbole'}
sorted_tags = sorted(lu_tag_counts.items(), key=lambda x: x[1], reverse=True)
tags, counts = zip(*sorted_tags)

# Map the old tags to new names
new_tags = [tag_mapping.get(tag, tag) for tag in tags]

ax1.bar(range(len(new_tags)), counts)
ax1.set_xticks(range(len(new_tags)))
ax1.set_xticklabels(new_tags, rotation=45, ha='right')
ax1.set_xlabel('Figure')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of Figures of Speech')

# Subplot 2: Lexical Unit Types (SingleWord, MWE, and Chunk)
# Combine ContiguousMWE and NonContiguousMWE under MWE
mwe_count = lu_type_counts['ContiguousMWE'] + lu_type_counts['NonContiguousMWE']
combined_counts = {
    'SingleWord': lu_type_counts['single_word'],
    'MWE': mwe_count,
    'Chunk': lu_type_counts['CHUNK']
}

lu_types = ['SingleWord', 'MWE', 'Chunk']
type_counts = [combined_counts[t] for t in lu_types]

ax2.pie(type_counts, labels=lu_types, autopct='%1.1f%%', startangle=90)
ax2.set_title('Proportion of Lexical Unit Types')
ax2.axis('equal')

# Adjust layout and add a main title
plt.tight_layout()
fig.suptitle('Lexical Unit Analysis', fontsize=16)
plt.subplots_adjust(top=0.9)

# Save the figure
output_file = os.path.join(OUTPUT_DIR, 'lexical_unit_analysis.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close(fig)  # Close the figure to free up memory

## Visualize attribute information

def categorize_attribute(attr_type):
    if attr_type in ['Conventionality', 'Directness']:
        return 'Metaphor'
    elif attr_type in ['Dimension', 'Degree']:
        return 'Hyperbole'
    return None

# Flatten the attributes and categorize them in one go
categorized_attributes = chain.from_iterable(
    ((categorize_attribute(attr.type), attr.value) 
     for attr in example.attributes 
     if categorize_attribute(attr.type) is not None)
    for example in examples
)

# Count the attributes
attribute_counts = defaultdict(Counter)
for category, value in categorized_attributes:
    attribute_counts[category][value] += 1

# Plot the distribution of attributes

def plot_attribute_counts(attribute_counts):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Metaphor plot
    metaphor_data = attribute_counts['Metaphor']
    conv_data = [metaphor_data.get('CNV', 0), metaphor_data.get('NOV', 0)]
    dir_data = [metaphor_data.get('DIR', 0), metaphor_data.get('IND', 0)]
    
    x = np.arange(2)
    width = 0.35
    
    ax1.bar(x - width/2, conv_data, width, label='Conventionality')
    ax1.bar(x + width/2, dir_data, width, label='Directness')
    
    ax1.set_ylabel('Counts')
    ax1.set_title('Metaphor Attributes')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Conventional/Direct', 'Novel/Indirect'])
    ax1.legend()
    
    for i, v in enumerate(conv_data):
        ax1.text(i - width/2, v, str(v), ha='center', va='bottom')
    for i, v in enumerate(dir_data):
        ax1.text(i + width/2, v, str(v), ha='center', va='bottom')
    
    # Hyperbole plot
    hyperbole_data = attribute_counts['Hyperbole']
    dim_data = [hyperbole_data.get('QNT', 0), hyperbole_data.get('QLT', 0)]
    deg_data = [hyperbole_data.get('1', 0), hyperbole_data.get('2', 0), hyperbole_data.get('3', 0)]
    
    x_dim = np.arange(2)
    x_deg = np.arange(3)
    
    ax2.bar(x_dim - width/2, dim_data, width, label='Dimension')
    ax2.bar(x_deg + width/2, deg_data, width, label='Degree')
    
    ax2.set_ylabel('Counts')
    ax2.set_title('Hyperbole Attributes')
    ax2.set_xticks(np.arange(3))
    ax2.set_xticklabels(['Quantitative/Degree 1', 'Qualitative/Degree 2', 'Degree 3'])
    ax2.legend()
    
    for i, v in enumerate(dim_data):
        ax2.text(i - width/2, v, str(v), ha='center', va='bottom')
    for i, v in enumerate(deg_data):
        ax2.text(i + width/2, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(OUTPUT_DIR, 'attribute_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory

plot_attribute_counts(attribute_counts)

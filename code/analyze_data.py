import os
import shutil
from config import *
import peek
import pandas as pd
import numpy as np
import json
from config import *
from parse_data import ExtendedExample, ExtendedBratParser, LexicalUnit, Span, Relation, Attribute
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from itertools import chain
from dataclasses import dataclass
from matplotlib import pyplot as plt


## Declare variables
handler = CollectionHandler(ANNOTATED_DIR)

all_corpora = handler.get_collections()
print("All annotated corpora:", all_corpora)
MAIN_PATH = handler.get_collection_path(os.path.join(ANNOTATED_DIR, 'main_collection/main_final'))
agr_path = handler.get_collection_path("agr_collection")
subcollections = handler.get_subcollections("agr_collection")
OUTPUT_DIR = TEMP_DIR

## Read in the corpus and perform some basic operations
main_corpus = peek.AnnCorpus(MAIN_PATH, txt=True)
doc = main_corpus.get_random_doc()
print("Random corpus document from the main collection:")
print()
print(doc.anns.items())

## Get a list of empty (un-annotated) documents and move non-empty documents to a new subcollection
#empty_docs = main_corpus.get_empty_files()
#annotated_docs = [doc.path for doc in main_corpus.docs if doc.path not in empty_docs]

## Create new subcollection called 'main_final'
#new_subcollection_path = os.path.join(main_path, "main_final")
#os.makedirs(new_subcollection_path, exist_ok=True)

# Copy non-empty documents (meaning the ann-files in annotated_docs and corresponding txt-files) to the new subcollection
#for ann_path in annotated_docs:
    # Get the corresponding txt-file from the ann-file (all ann_paths ends with .ann)
    #txt_path = ann_path[:-3] + 'txt'
    # Copy the ann-file and txt-file to the new subcollection in case they don't already exist
    #if not os.path.exists(os.path.join(new_subcollection_path, os.path.basename(ann_path))):
        #shutil.copy(ann_path, new_subcollection_path)
    #if not os.path.exists(os.path.join(new_subcollection_path, os.path.basename(txt_path))):
        #shutil.copy(txt_path, new_subcollection_path)

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

### Gather advanced stats ###

# Get lexical units info:
lexical_units = [lu for example in examples for lu in example.lexical_units]
lu_counts = len(lexical_units)

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
mtp_lu_counts = Counter(lu.text for lu in mtp_lus)
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

## Create an all_lexical_units dictionary with document_ID as key, and a list of lexical units as values.
all_lus = {
    example.id: [(lu.text, lu.spans, lu.type, lu.tag) for lu in example.lexical_units]
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

# Get lexical units for document with ID 9166397
doc_id = str(9166397)
doc_lus = all_lus[doc_id]

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
    categories = list(attribute_counts.keys())
    attributes = set().union(*[set(counts.keys()) for counts in attribute_counts.values()])
    
    x = np.arange(len(attributes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, category in enumerate(categories):
        counts = [attribute_counts[category].get(attr, 0) for attr in attributes]
        ax.bar(x + i*width, counts, width, label=category)
    
    ax.set_ylabel('Counts')
    ax.set_title('Attribute Counts for Metaphor and Hyperbole')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(attributes)
    ax.legend()
    
    plt.tight_layout()
    
    # Save the figure instead of showing it
    output_file = os.path.join(OUTPUT_DIR, 'attribute_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory

# Call the function with your data
plot_attribute_counts(attribute_counts)

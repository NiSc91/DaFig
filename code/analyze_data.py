import pdb
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
from nltk.tokenize import sent_tokenize
from collections import Counter, defaultdict
from itertools import chain, combinations
from dataclasses import dataclass
from matplotlib import pyplot as plt
import seaborn as sns
from pygamma_agreement import Continuum, CombinedCategoricalDissimilarity
from pyannote.core import Segment
from pygamma_agreement import show_alignment
import spacy
import lemmy

# Variables
handler = CollectionHandler(CORPORA_DIR)
all_corpora = handler.get_collections()
print("All annotated corpora:", all_corpora)

# Create path for main corpus
MAIN_PATH = handler.get_collection_path(os.path.join(CORPORA_DIR,'main'))
# Create paths for the agreement corpora
CORPUS_NAMES = ['main', 'agr1', 'agr2', 'agr3', 'agr_combined', 'consensus']
CORPUS_PATHS = {f"{name.upper()}_PATH": handler.get_collection_path(os.path.join(CORPORA_DIR, name)) for name in CORPUS_NAMES}
OUTPUT_DIR = TEMP_DIR

# Check if the output directory exists, if not, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Create a lambda function to get ann paths
get_ann_path = lambda base_path, ann_folder: os.path.join(base_path, ann_folder)

# Create a dictionary with the paths to the ann folders for each corpus except for main
base_paths = {name: handler.get_collection_path(os.path.join(CORPORA_DIR, name)) for name in CORPUS_NAMES[1:]}
ann_paths = {f"{name.upper()}_ANN1_PATH": get_ann_path(base_path, 'ann1') for name, base_path in base_paths.items()}
ann_paths.update({f"{name.upper()}_ANN2_PATH": get_ann_path(base_path, 'ann2') for name, base_path in base_paths.items()})

### Analysis on training corpus ###

## Read in the main corpus and perform some basic operations
main_corpus = peek.AnnCorpus(MAIN_PATH, txt=True)
doc = main_corpus.get_random_doc()
print("Random corpus document from the main collection:")
print()
print(doc.anns.items())

# Create span class to handle spans
@dataclass(frozen=True)
class Span:
    start: int
    end: int

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

# This has been done across all corpora
#misaligned_main = write_misaligned_spans(MAIN_PATH, OUTPUT_DIR)
#misaligned_agr1 = write_misaligned_spans(AGR1_PATH, OUTPUT_DIR)
#misaligned_agr2 = write_misaligned_spans(AGR2_PATH, OUTPUT_DIR)
#misaligned_agr3 = write_misaligned_spans(AGR3_PATH, OUTPUT_DIR)
#misaligned_agr_final = write_misaligned_spans(AGR_FINAL_PATH, OUTPUT_DIR)
#misaligned_consensus_agr2 = write_misaligned_spans(CONSENSUS_PATH, OUTPUT_DIR)

### Gather advanced stats ###

# Get lexical units info:
lexical_units = [lu for example in examples for lu in example.lexical_units]
lu_counts = len(lexical_units)
# Get the number of tokens for all lexical units
token_counts = sum(len(lu.mention.split()) for lu in lexical_units)

# Get tag info
lu_tag_counts = Counter(lu.tag for lu in lexical_units)

# Get MWE and chunk info:
mwe_chunk_counts = sum(len(lu.spans) > 1 for lu in lexical_units)
lu_type_counts = Counter(lu.type for lu in lexical_units)

# Get the types of lexical units, single words, MWEs (contiguous and non-contiguous), and chunks, across each of the tags (MTP, HPB, etc.)
lu_type_tag_counts = Counter((lu.type, lu.tag) for lu in lexical_units)
# Print out the counts for each tag
for tag, counts in lu_type_tag_counts.items():
    print(f"{tag}: {counts}")

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

# Create an all_lexical_units dictionary with document_ID as key, and a list of lexical units as values.
all_lus = {
    example.id: [(lu.mention, lu.spans, lu.type, lu.tag) for lu in example.lexical_units]
    for example in examples
}

# Create a new dictionary with document_ID as key, and a list of entities which have different tags but same or overlapping spans, as values.
lu_overlaps = defaultdict(set)

for doc_id, lus in all_lus.items():
    # Use combinations to compare each pair of LUs only once
    for (lu1, spans1, type1, tag1), (lu2, spans2, type2, tag2) in combinations(lus, 2):
        if tag1 != tag2 and spans1 == spans2:
            # Create tuples for each LU info, converting spans to a hashable type
            lu_info1 = (lu1, tuple((s.start, s.end) for s in spans1), type1, tag1)
            lu_info2 = (lu2, tuple((s.start, s.end) for s in spans2), 
                        type2, tag2)
            # Use a frozenset to ensure uniqueness and allow it to be added to a set
            overlap = frozenset([lu_info1, lu_info2])
            lu_overlaps[doc_id].add(overlap)

# Convert sets back to lists for consistency with your original output
lu_overlaps = {k: [tuple(o) for o in v] for k, v in lu_overlaps.items()}

# Output the results to a json-file and save to OUTPUT_DIR as lu_overlaps.json
with open(os.path.join(OUTPUT_DIR, 'lu_overlaps.json'), 'w', encoding='utf-8') as f:
    json.dump(lu_overlaps, f, ensure_ascii=False, indent=2)

### Extract sentences with figurative tag information

def get_figurative_sentences(lus_dict, texts_dict, fig_tags):
    """
    Extract sentences containing figurative language and their associated tags using NLTK.
    
    Args:
        lus_dict (dict): Dictionary of lexical units with their spans and tags
        texts_dict (dict): Dictionary of full texts
        fig_tags (list): List of figurative tags to look for (e.g., ['MTP', 'HPB', 'BIR'])
    
    Returns:
        pd.DataFrame: DataFrame with columns ['doc_id', 'sentence', 'tagged_word', 'tag']
    """
    results = []
    
    for doc_id, text in texts_dict.items():
        if doc_id not in lus_dict:
            continue
            
        # Split text into sentences using NLTK and newlines
        sentences = [sent.strip() 
                    for line in text.split('\n')
                    for sent in sent_tokenize(line, language='danish')]
        
        # Calculate sentence spans
        current_pos = 0
        sent_spans = []
        
        for sent in sentences:
            # Find the exact position of the sentence in the original text
            sent_start = text.find(sent, current_pos)
            sent_end = sent_start + len(sent)
            sent_spans.append((sent_start, sent_end, sent))
            current_pos = sent_end        
            
        # Process each lexical unit
        for lu in lus_dict[doc_id]:
            mention, spans, type, tag = lu
            
            # Skip if tag not in requested figurative tags
            if tag not in fig_tags:
                continue
            
            # Get the start position of the first span
            lu_start = spans[0].start
            
            # Find which sentence contains this lexical unit
            for sent_start, sent_end, sentence in sent_spans:
                if sent_start <= lu_start < sent_end:
                    results.append({
                        'doc_id': doc_id,
                        'tagged_word': mention.lower(),
                        'sentence': sentence.strip(),
                        'tag': tag,
                        'type': type
                    })
                    break
    
    return pd.DataFrame(results)

# Find metaphorical sentences (tagged MTP)
mtp_df = get_figurative_sentences(all_lus, all_texts, ['MTP'])
# Sort the metaphorical sentences by the tagged_word column
mtp_df = mtp_df.sort_values('tagged_word')

# Find hyperbolic sentences (tagged HPB)
hpb_df = get_figurative_sentences(all_lus, all_texts, ['HPB'])
# Sort the hyperbolic sentences by the tagged_word column
hpb_df = hpb_df.sort_values('tagged_word')

## Lemmatize metaphors using spaCy
nlp = spacy.load("da_core_news_md")  # Load the Danish model

# Process the metaphors in mtp_df
mtp_df['lemma'] = mtp_df['tagged_word'].apply(lambda word: nlp(word)[0].lemma_)

## Output the lemmatized metaphors to a csv file
mtp_df.to_csv(os.path.join(OUTPUT_DIR,'mtp_sentences_lemmatized.csv'), encoding='utf-8', index=False)

# Pull metaphor list from txt-file
metaphor_list = []
with open(os.path.join(DATA_DIR,'metaphor_lemma_list.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        # Make sure that numbers and other non/alphanumeric characters are removed before adding it to the list
        metaphor_list.append(re.sub(r'[^a-zA-Z0-9 ]+', '', line.strip()))

## Create a new dataframe consisting of the subset of lemmas in mtp_df which also exist in the metaphor list
lemmatized_mtp_df = mtp_df[mtp_df['lemma'].isin(metaphor_list)]

## Save the results to a csv file
lemmatized_mtp_df.to_csv(os.path.join(OUTPUT_DIR,'mtp_sentences_from_DDO.csv'), encoding='utf-8', index=False)

### Visualize attribute information

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

# Print the attribute counts
for category, counter in attribute_counts.items():
    print(f"{category} Attribute Counts:")
    print(counter)
    print()

### Analyses on agreement corpora ###

def create_combined_annotations(annotator1, annotator2):
    """
    Create a nested combined lexical units dictionary organized by label and doc_id.
    Structure: {label: {doc_id: [annotator1_annotations, annotator2_annotations]}}
    """
    labels = ['MTP', 'HPB', 'VIR', 'WIDLII']
    combined_annotations = {label: {} for label in labels}
    
    for annotator, corpus in [('annotator1', annotator1), ('annotator2', annotator2)]:
        for example in corpus:
            doc_id = example.id
            for lu in example.lexical_units:
                label = lu.tag
                if label not in labels:
                    continue  # Skip labels not in the predefined list
                if doc_id not in combined_annotations[label]:
                    combined_annotations[label][doc_id] = ([], [])
                if annotator == 'annotator1':
                    combined_annotations[label][doc_id][0].append((lu.mention, lu.spans, lu.type, lu.tag))
                else:
                    combined_annotations[label][doc_id][1].append((lu.mention, lu.spans, lu.type, lu.tag))
    
    return combined_annotations

def convert_to_gamma_format(annotations_dict):
    """
    Convert annotations to a gamma Continuum format.
    """
    continuum = Continuum()
    
    for doc_id, annotator_pairs in annotations_dict.items():
        # Process each annotator's annotations
        for annotator_idx, annotations in enumerate(annotator_pairs, 1):
            annotator_name = f"Annotator{annotator_idx}"
            
            # Process each annotation tuple (mention, spans, type, tag)
            for _, spans, _, category in annotations:
                for span in spans:
                    # Convert character positions to "time-like" segments
                    # Using character positions as if they were timestamps
                    segment = Segment(span.start, span.end)
                    continuum.add(annotator_name, segment, category)
    
    return continuum

def calculate_gamma_agreement(ann1_path, ann2_path):
    """
    Calculate overall and per-label gamma agreements between two annotators.
    
    Args:
        ann1_path (str): Path to annotator 1's annotations.
        ann2_path (str): Path to annotator 2's annotations.
        
    Returns:
        dict: A dictionary containing overall gamma and per-label gamma scores.
    """
    # Load the annotations for two annotators
    brat1 = ExtendedBratParser(input_dir=ann1_path, error="ignore")
    ann1_corpus = list(brat1.parse(ann1_path))
    
    brat2 = ExtendedBratParser(input_dir=ann2_path, error="ignore")
    ann2_corpus = list(brat2.parse(ann2_path))
    
    # Create a nested combined annotations dictionary per label
    combined_annotations = create_combined_annotations(ann1_corpus, ann2_corpus)
    
    # Initialize results dictionary
    agreement_results = {}
    
    # Calculate overall gamma agreement across all labels
    # Merge all labels into a single annotations dictionary
    overall_annotations = {}
    for label_dict in combined_annotations.values():
        for doc_id, annotator_pairs in label_dict.items():
            if doc_id not in overall_annotations:
                overall_annotations[doc_id] = ([], [])
            overall_annotations[doc_id][0].extend(annotator_pairs[0])
            overall_annotations[doc_id][1].extend(annotator_pairs[1])
    
    overall_continuum = convert_to_gamma_format(overall_annotations)
    dissim = CombinedCategoricalDissimilarity(delta_empty=1, alpha=3, beta=1)
    overall_gamma_agreement = overall_continuum.compute_gamma(dissim, precision_level=0.02)
    agreement_results['Overall'] = overall_gamma_agreement.gamma
    
    # Calculate gamma agreement per label
    for label, label_annotations in combined_annotations.items():
        if not label_annotations:
            agreement_results[label] = None  # No data for this label
            continue
        
        label_continuum = convert_to_gamma_format(label_annotations)
        # If AssertionError, handle the error and replace with 0
        try:
            label_gamma_agreement = label_continuum.compute_gamma(dissim)
        except AssertionError as e:
            print(f"AssertionError in {label}: {e}")
            agreement_results[label] = 0
            continue
        
        agreement_results[label] = label_gamma_agreement.gamma
        label_gamma_agreement = label_continuum.compute_gamma(dissim)
        agreement_results[label] = label_gamma_agreement.gamma
    
    return agreement_results

### Create gamma agreement report for multiple corpora
# Initialize a list to collect results
gamma_results_list = []

# Define annotation stages and their corresponding paths
annotation_stages = {
    'AGR1\n(5h Annotation)': ('AGR1_ANN1_PATH', 'AGR1_ANN2_PATH'),
    'AGR2\n(40h Annotation)': ('AGR2_ANN1_PATH', 'AGR2_ANN2_PATH'),
    'Consensus\n(AGR2 Discussion)': ('CONSENSUS_ANN1_PATH', 'CONSENSUS_ANN2_PATH'),
    'AGR3\n(Final Stage)': ('AGR3_ANN1_PATH', 'AGR3_ANN2_PATH')
}

# Iterate through each annotation stage and calculate gamma agreements
for stage_label, (ann1_key, ann2_key) in annotation_stages.items():
    ann1_path = ann_paths[ann1_key]
    ann2_path = ann_paths[ann2_key]
    gamma_agreement = calculate_gamma_agreement(ann1_path, ann2_path)
    
    # Add stage label to each result
    for metric, score in gamma_agreement.items():
        gamma_results_list.append({
            'Annotation Stage': stage_label,
            'Metric': metric,
            'Gamma Score': score
        })
    
    # Optional: Print progress
    print(f"Gamma Agreement for {stage_label}: {gamma_agreement}")

# Convert the list of results into a pandas DataFrame
gamma_df = pd.DataFrame(gamma_results_list)

# Drop rows where gamma score is NaN, and exclude the 'VIR' label if present
gamma_df_clean = gamma_df.dropna(subset=['Gamma Score'])
gamma_df_clean = gamma_df_clean[gamma_df_clean['Metric']!= 'VIR']

# Verify the cleaned DataFrame
print(gamma_df_clean.head())

### Visualize gamma agreement progression
# Set seaborn theme for publication-quality graphics
sns.set(style='whitegrid', font_scale=1.2)

# Define the order of metrics to ensure consistent coloring
metric_order = ['Overall', 'MTP', 'HPB', 'WIDLII']

# Define a color palette
palette = sns.color_palette("Set2", n_colors=len(metric_order))

# Initialize the matplotlib figure
plt.figure(figsize=(14, 8))

# Create the bar plot using the cleaned DataFrame
sns.barplot(
    data=gamma_df_clean,
    x='Annotation Stage',
    y='Gamma Score',
    hue='Metric',
    palette=palette,
    order=sorted(gamma_df_clean['Annotation Stage'].unique()),  # Adjust as needed
    hue_order=metric_order
)

# Customize the plot
plt.title('Inter-Annotator Gamma Agreement Across Annotation Stages and Labels', fontsize=16, pad=20)
plt.xlabel('Annotation Stage', fontsize=14)
plt.ylabel('Gamma Agreement Score', fontsize=14)
plt.ylim(0, 1.05)  # Adjust if needed based on your data

plt.legend(title='Metric', fontsize=12, title_fontsize=13, loc='upper left', bbox_to_anchor=(1, 1))

# Add value labels on top of each bar, skipping non-finite heights
for p in plt.gca().patches:
    height = p.get_height()
    if np.isfinite(height):
        plt.gca().text(
            p.get_x() + p.get_width() / 2., 
            height + 0.01, 
            f'{height:.2f}', 
            ha='center', 
            va='bottom',
            fontsize=10
        )

# Adjust layout to accommodate the legend
plt.tight_layout()

# Save the figure
output_path = os.path.join(OUTPUT_DIR, 'gamma_agreement_grouped_bar.png')   
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Save gamma_df to a CSV file
gamma_df.to_csv(os.path.join(OUTPUT_DIR, 'gamma_agreement_scores.csv'), index=False)
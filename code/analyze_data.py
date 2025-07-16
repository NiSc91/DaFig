import pdb
import logging
import os
import shutil
from config import *
import peek
#from nltk import RegexpTokenizer
import re
import pandas as pd
import numpy as np
import json
from parse_data import ExtendedExample, ExtendedBratParser, LexicalUnit, Span, Entity, Relation, Attribute
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from collections import Counter, defaultdict
from itertools import chain, combinations
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

# Create paths for the corpora
CORPUS_NAMES = ['main', 'agr1', 'agr2', 'agr3', 'agr_combined', 'consensus', 'reanno']
CORPUS_PATHS = {f"{name.upper()}_PATH": handler.get_collection_path(os.path.join(CORPORA_DIR, name)) for name in CORPUS_NAMES}
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'analysis')

# Check if the output directory exists, if not, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Create path for main corpus
MAIN_PATH = CORPUS_PATHS['MAIN_PATH']

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

### Find misaligned spans, i.e. instances where an entity span/mention does not correspond to a token in the text ###

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
from collections import Counter, defaultdict
from itertools import combinations

def gather_corpus_stats(corpus_path):
    # Initialize the ExtendedBratParser
    brat = ExtendedBratParser(input_dir=corpus_path, error="ignore")
    
    # Parse the corpus
    examples = list(brat.parse(corpus_path))
    
    # Initialize statistics
    stats = {
        'total_lexical_units': 0,
        'token_counts': 0,
        'lu_tag_counts': Counter(),
        'mwe_chunk_counts': 0,
        'lu_type_counts': Counter(),
        'lu_type_tag_counts': Counter(),
        'mtp_lu_counts': Counter(),
        'hpb_lu_counts': Counter(),  # Added for hyperboles
        'total_tokens': 0,
        'overlapping_spans': defaultdict(set)
    }
    
    # Gather lexical units
    lexical_units = [lu for example in examples for lu in example.lexical_units]
    stats['total_lexical_units'] = len(lexical_units)
    
    # Process lexical units
    for lu in lexical_units:
        stats['token_counts'] += len(lu.mention.split())
        stats['lu_tag_counts'][lu.tag] += 1
        stats['mwe_chunk_counts'] += (len(lu.spans) > 1)
        stats['lu_type_counts'][lu.type] += 1
        stats['lu_type_tag_counts'][(lu.type, lu.tag)] += 1
        
        if lu.tag == 'MTP':
            stats['mtp_lu_counts'][lu.mention] += 1
        elif lu.tag == 'HPB':  # Added for hyperboles
            stats['hpb_lu_counts'][lu.mention] += 1
    
    # Count total tokens in the corpus
    stats['total_tokens'] = sum(len(example.text.split()) for example in examples)
    
    # Create dictionaries for texts and lexical units
    all_texts = {example.id: example.text for example in examples}
    all_lus = {
        example.id: [(lu.mention, lu.spans, lu.type, lu.tag) for lu in example.lexical_units]
        for example in examples
    }
    
    # Find overlapping spans
    for doc_id, lus in all_lus.items():
        for (lu1, spans1, type1, tag1), (lu2, spans2, type2, tag2) in combinations(lus, 2):
            if tag1 != tag2 and spans1 == spans2:
                lu_info1 = (lu1, tuple((s.start, s.end) for s in spans1), type1, tag1)
                lu_info2 = (lu2, tuple((s.start, s.end) for s in spans2), type2, tag2)
                overlap = frozenset([lu_info1, lu_info2])
                stats['overlapping_spans'][doc_id].add(overlap)
    
    # Convert sets to lists for JSON serialization
    stats['overlapping_spans'] = {k: list(v) for k, v in stats['overlapping_spans'].items()}
    
    return stats

# Example usage:
# corpus_stats = gather_corpus_stats(MAIN_PATH)
# print(corpus_stats)

### Visualize attribute information

"""def categorize_attribute(attr_type):
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
"""

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
gamma_df.to_csv(os.path.join(OUTPUT_DIR, 'gamma_agreement_scores.csv'), index=False)### Extract sentences from data ###

### Extract sentences containing figurative language ###

def get_figurative_sentences(corpus_path, fig_tags):
    """
    Extract and lemmatize sentences containing figurative language from a given corpus.
    Include attribute types and values based on the tag of the lexical unit.
    
    Args:
        corpus_path (str): Path to the corpus
        fig_tags (list): List of figurative tags to look for (e.g., ['MTP', 'HPB'])
    
    Returns:
        pd.DataFrame: DataFrame with columns ['doc_id', 'sentence', 'tagged_word', 'tag', 'type', 'lemma', 'pos']
                      and additional columns for attributes based on the tag
    """
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize the ExtendedBratParser
    brat = ExtendedBratParser(input_dir=corpus_path, error="ignore")
    
    # Parse the corpus
    examples = list(brat.parse(corpus_path))
    
    results = []
    
    # Load spaCy model
    nlp = spacy.load("da_core_news_md")  # Load the Danish model
    
    for example in examples:
        doc_id = example.id
        text = example.text
        
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
        for lu in example.lexical_units:
            # Skip if tag not in requested figurative tags
            if lu.tag not in fig_tags:
                continue
            
            # Get the start position of the first span
            lu_start = lu.spans[0].start
            
            # Find which sentence contains this lexical unit
            for sent_start, sent_end, sentence in sent_spans:
                if sent_start <= lu_start < sent_end:
                    # Process the tagged word with spaCy
                    doc = nlp(lu.mention.lower())
                    lemma = doc[0].lemma_
                    pos = doc[0].pos_
                    
                    result = {
                        'doc_id': doc_id,
                        'tagged_word': lu.mention.lower(),
                        'sentence': sentence.strip(),
                        'tag': lu.tag,
                        'type': lu.type,
                        'lemma': lemma,
                        'pos': pos
                    }
                    
                    # Process attributes based on the tag
                    if lu.tag == 'MTP':
                        result['Conventionality'] = lu.attributes.get('Conventionality', [None])[0]
                        result['Directness'] = lu.attributes.get('Directness', [None])[0]
                    elif lu.tag == 'HPB':
                        result['Dimension'] = lu.attributes.get('Dimension', [None])[0]
                        result['Degree'] = lu.attributes.get('Degree', [None])[0]
                    
                    results.append(result)
                    break
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    return df 

def process_corpus_data(corpus_info, fig_tags=['MTP', 'HPB']):
    """
    Process both single and multi-annotated corpora.
    
    Args:
        corpus_info (dict): A dictionary containing corpus information.
            For single-annotated: {'name': 'CORPUS_NAME', 'path': 'CORPUS_PATH', 'type': 'single'}
            For multi-annotated: {'name': 'CORPUS_NAME', 'path': ('ANN1_PATH', 'ANN2_PATH'), 'type': 'multi'}
        fig_tags (list): List of figurative tags to process.
    """
    if corpus_info['type'] == 'single':
        for tag in fig_tags:
            df = get_figurative_sentences(corpus_info['path'], [tag])
            #df = df.sort_values('tagged_word')
            output_filename = f'{tag.lower()}_sentences_lemmatized_{corpus_info["name"].lower()}.csv'
            df.to_csv(os.path.join(OUTPUT_DIR, output_filename), encoding='utf-8', index=False)
    elif corpus_info['type'] == 'multi':
        ann1_path, ann2_path = corpus_info['path']
        for tag in fig_tags:
            # Process annotations from both annotators
            ann1_df = get_figurative_sentences(ann1_path, [tag])
            ann2_df = get_figurative_sentences(ann2_path, [tag])

            # Create a unique identifier for each sentence-tagged_word pair
            ann1_df['sent_word'] = ann1_df['sentence'] + '|' + ann1_df['tagged_word']
            ann2_df['sent_word'] = ann2_df['sentence'] + '|' + ann2_df['tagged_word']

            # Merge the dataframes
            merged_df = pd.merge(ann1_df, ann2_df, 
                                 on=['doc_id', 'sentence', 'sent_word'], 
                                 how='inner', 
                                 suffixes=('_ann1', '_ann2'))

            # Keep only the rows where both annotators have annotated
            merged_df = merged_df.dropna(subset=['lemma_ann1', 'lemma_ann2'])

            # Combine annotations
            for col in ['tagged_word', 'lemma', 'tag', 'type']:
                merged_df[col] = merged_df[f'{col}_ann1']

            # Process attributes
            if tag == 'MTP':
                attr_cols = ['Conventionality', 'Directness']
            elif tag == 'HPB':
                attr_cols = ['Dimension', 'Degree']
            else:
                attr_cols = []

            for attr in attr_cols:
                merged_df[f'{attr}_agreement'] = merged_df[f'{attr}_ann1'] == merged_df[f'{attr}_ann2']
                merged_df[attr] = merged_df[f'{attr}_ann1']

            # Select final columns
            final_cols = ['doc_id', 'sentence', 'tagged_word', 'lemma', 'tag', 'type'] + attr_cols
            if attr_cols:
                final_cols += [f'{attr}_agreement' for attr in attr_cols]

            df = merged_df[final_cols]

            #df = df.sort_values(['sentence', 'tagged_word'])
            output_filename = f'{tag.lower()}_sentences_lemmatized_{corpus_info["name"].lower()}.csv'
            df.to_csv(os.path.join(OUTPUT_DIR, output_filename), encoding='utf-8', index=False)
    
    print(f"Processed {corpus_info['name']}")

# List of corpora to process
corpora_to_process = [
    {'name': 'MAIN', 'path': CORPUS_PATHS['MAIN_PATH'], 'type': 'single'},
    {'name': 'REANNO', 'path': CORPUS_PATHS['REANNO_PATH'], 'type': 'single'},
    {'name': 'CONSENSUS', 'path': (ann_paths['CONSENSUS_ANN1_PATH'], ann_paths['CONSENSUS_ANN2_PATH']), 'type': 'multi'}
]

# Process all corpora
for corpus in corpora_to_process:
    process_corpus_data(corpus)

print("Processed all datasets")
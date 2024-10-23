import pdb
import re
import json
from dataclasses import dataclass
from typing import List, Tuple, NamedTuple, Dict
from parse_data import ExtendedExample, ExtendedBratParser, LexicalUnit, Span, Relation, Attribute
from config import *

## Declare variables
handler = CollectionHandler(CORPORA_DIR)
all_corpora = handler.get_collections()
print("All annotated corpora:", all_corpora)

# Create path for main corpus
MAIN_PATH = handler.get_collection_path(os.path.join(CORPORA_DIR,'main'))
# Create paths for the agreement corpora
AGR_NAMES = ['main', 'agr1', 'agr2', 'agr3', 'agr_combined', 'consensus']
AGR_PATHS = {f"{name.upper()}_PATH": handler.get_collection_path(os.path.join(CORPORA_DIR, name)) for name in AGR_NAMES}
OUTPUT_DIR = TEMP_DIR

# Create a lambda function to get ann paths
get_ann_path = lambda base_path, ann_folder: os.path.join(base_path, ann_folder)

# Create a dictionary with the paths to the ann folders for each corpus except for main
base_paths = {name: handler.get_collection_path(os.path.join(CORPORA_DIR, name)) for name in AGR_NAMES}
ann_paths = {f"{name.upper()}_ANN1_PATH": get_ann_path(base_path, 'ann1') for name, base_path in AGR_PATHS.items()}
ann_paths.update({f"{name.upper()}_ANN2_PATH": get_ann_path(base_path, 'ann2') for name, base_path in AGR_PATHS.items()})

### Implement the necessary classes and function to implement the extended BIO scheme ###

@dataclass
class Span:
    start: int
    end: int

@dataclass
class LexicalUnit:
    mention: str
    spans: List[Span]
    unit_type: str
    fig_type: str

class   FigurativeLanguageTagger:
    def __init__(self):
        self.valid_tags = {'O', 'B', 'I_', 'I~', 'o', 'b', 'i_', 'i~'}
        self.figurative_tags = {'MTP', 'VIR', 'HPB'}

    def tokenize(self, text):
        return re.findall(r'\w+|[^\w\s]', text)

    def map_char_to_token_spans(self, text, tokens, char_spans):
        token_spans = []
        char_index = 0
        for token_index, token in enumerate(tokens):
            token_start = text.index(token, char_index)
            token_end = token_start + len(token)
            for char_span in char_spans:
                if (char_span.start >= token_start and char_span.start < token_end) or \
                   (char_span.end > token_start and char_span.end <= token_end) or \
                   (char_span.start <= token_start and char_span.end >= token_end):
                    token_spans.append(token_index)
                    break
            char_index = token_end
        return token_spans

    def tag_document(self, text: str, lexical_units: List[LexicalUnit]):
        tokens = self.tokenize(text)
        tagged_tokens = ['O'] * len(tokens)
        
        for unit in lexical_units:
            unit = LexicalUnit(unit[0], unit[1], unit[2], unit[3])
            token_spans = self.map_char_to_token_spans(text, tokens, unit.spans)
            
            fig_tags = unit.fig_type.split('|')
            fig_tag = '-' + '|'.join(fig_tags) if fig_tags else ''
            
            if unit.unit_type == 'single_word':
                if token_spans:
                    tagged_tokens[token_spans[0]] = f'B{fig_tag}'
            elif unit.unit_type in ['ContiguousMWE', 'NonContiguousMWE', 'CHUNK']:
                is_weak = unit.unit_type == 'CHUNK' # Always contiguous
                is_non_contiguous = unit.unit_type == 'NonContiguousMWE'
                
                for i, token_index in enumerate(token_spans):
                    if i == 0:
                        tagged_tokens[token_index] = f'B{fig_tag}'
                    else:
                        if not is_weak:
                            tagged_tokens[token_index] = f'I_{fig_tag}'
                        else:
                            tagged_tokens[token_index] = f'I~{fig_tag}'
                    
                    # Handle gaps for non-contiguous MWEs
                    if is_non_contiguous and i < len(token_spans) - 1:
                        next_index = token_spans[i + 1]
                        for gap_index in range(token_index + 1, next_index):
                            current_tag = tagged_tokens[gap_index]
                            if current_tag == 'O':
                                tagged_tokens[gap_index] = 'o'
                            elif current_tag.startswith('B'):
                                tagged_tokens[gap_index] = 'b' + current_tag[1:]
                            elif current_tag.startswith('I_'):
                                tagged_tokens[gap_index] = 'i_' + current_tag[2:]
                            elif current_tag.startswith('I~'):
                                tagged_tokens[gap_index] = 'i~' + current_tag[2:]
        
        return list(zip(tokens, tagged_tokens))

    def print_tagged(self, tagged_tokens):
        for token, tag in tagged_tokens:
            print(f"{token}/{tag}", end=" ")
        print()

## Apply tagger to all documents in the main corpus

brat = ExtendedBratParser(input_dir=MAIN_PATH, error="ignore")
examples = list(brat.parse(MAIN_PATH))

## Extract example texts and their IDs in a dictionary for easy access
texts = {example.id: example.text for example in examples}

## Create an all_lexical_units dictionary with document_ID as key, and a list of lexical units as values.
all_lus = {
    example.id: [(lu.mention, lu.spans, lu.type, lu.tag) for lu in example.lexical_units]
    for example in examples}

# Create a figurative language tagger
tagger = FigurativeLanguageTagger()

# Process all documents
all_tagged_documents = {}

for doc_id, doc_text in texts.items():
    doc_lus = all_lus[doc_id]
    
    # Tag the document
    tagged_tokens = tagger.tag_document(doc_text, doc_lus)
    
    # Store the tagged document
    all_tagged_documents[doc_id] = tagged_tokens

# Save to a file
with open(os.path.join(OUTPUT_DIR, 'tagged_documents.json'), 'w', encoding='utf-8') as f:
    json.dump(all_tagged_documents, f, ensure_ascii=False, indent=2)

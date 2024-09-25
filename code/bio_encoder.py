import pdb
import re
from dataclasses import dataclass
from typing import List, Tuple
from parse_data import ExtendedExample, ExtendedBratParser, LexicalUnit, Span, Relation, Attribute
from config import *

## Declare variables
handler = CollectionHandler(ANNOTATED_DIR)
MAIN_PATH = handler.get_collection_path(os.path.join(ANNOTATED_DIR, 'main_collection/main_final'))
OUTPUT_DIR = TEMP_DIR

### Implement the necessary classes and function to implement the extended BIO scheme ###

@dataclass
class Span:
    start: int
    end: int

@dataclass
class LexicalUnit:
    text: str
    spans: List[Span]
    unit_type: str
    fig_type: str

class FigurativeLanguageTagger:
    def __init__(self):
        self.valid_tags = set(['O', 'B', 'I_', 'I~', 'o', 'b', 'i_', 'i~'])
        self.figurative_tags = set(['MTP', 'VIR', 'HPB'])

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
                    tagged_tokens[token_spans[0]] = f'O{fig_tag}'
            elif unit.unit_type in ['ContiguousMWE', 'NonContiguousMWE', 'Chunk']:
                is_weak = unit.unit_type == 'CHUNK'
                in_gap = False
                for i, token_index in enumerate(token_spans):
                    if i == 0:
                        tagged_tokens[token_index] = f'B{fig_tag}'
                    else:
                        if not is_weak:
                            tagged_tokens[token_index] = f'I_{fig_tag}' if not in_gap else f'i_{fig_tag}'
                        else:
                            tagged_tokens[token_index] = f'I~{fig_tag}' if not in_gap else f'i~{fig_tag}'
                    
                    # Check for gaps
                    if i < len(token_spans) - 1:
                        next_index = token_spans[i + 1]
                        if next_index - token_index > 1:
                            in_gap = True
                            for gap_index in range(token_index + 1, next_index):
                                tagged_tokens[gap_index] = '_'
                        else:
                            in_gap = False
        
        return list(zip(tokens, tagged_tokens))

    def print_tagged(self, tagged_tokens):
        for token, tag in tagged_tokens:
            print(f"{token}/{tag}", end=" ")
        print()

# Example usage

brat = ExtendedBratParser(input_dir=MAIN_PATH, error="ignore")
examples = list(brat.parse(MAIN_PATH))

## Extract example texts and their IDs in a dictionary for easy access
texts = {example.id: example.text for example in examples}

## Create an all_lexical_units dictionary with document_ID as key, and a list of lexical units as values.
all_lus = {
    example.id: [(lu.text, lu.spans, lu.type, lu.tag) for lu in example.lexical_units]
    for example in examples}

# Get a single document
doc_id = str(9166397)
doc_text = texts[doc_id]
doc_lus = all_lus[doc_id]

# Create a figurative language tagger
tagger = FigurativeLanguageTagger()

# Tag the document
pdb.set_trace()
tagged_tokens = tagger.tag_document(doc_text, doc_lus)
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
        # Simple tokenization by splitting on whitespace and punctuation
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
            token_spans = self.map_char_to_token_spans(text, tokens, unit.spans)
            
            if unit.unit_type == 'single_word':
                if token_spans:
                    tagged_tokens[token_spans[0]] = f'B-{unit.fig_type}'
            elif unit.unit_type in ['ContiguousMWE', 'NonContiguousMWE', 'Chunk']:
                for i, token_index in enumerate(token_spans):
                    if i == 0:
                        tagged_tokens[token_index] = f'B-{unit.fig_type}'
                    else:
                        tagged_tokens[token_index] = f'I{"_" if unit.unit_type == "ContiguousMWE" else "~"}-{unit.fig_type}'
        
        return list(zip(tokens, tagged_tokens))

    def print_tagged(self, tagged_tokens):
        for token, tag in tagged_tokens:
            print(f"{token}/{tag}", end=" ")
        print()

# Example usage

brat = ExtendedBratParser(input_dir=MAIN_PATH, error="ignore")
examples = list(brat.parse(MAIN_PATH))

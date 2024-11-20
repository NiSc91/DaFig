import re
import json
from dataclasses import dataclass
from typing import List, Tuple, NamedTuple, Dict
from parse_data import ExtendedExample, ExtendedBratParser, LexicalUnit, Span, Relation, Attribute
from nltk.tokenize import word_tokenize
from config import *

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

class   OldFigurativeLanguageTagger:
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

### Simplified tagging scheme ###

class FigurativeLanguageTagger:
    def __init__(self, tagging_scheme='joint'):
        self.tagging_scheme = tagging_scheme
        # Basic BIO tags
        self.valid_tags = {
            'O',    # Outside
            'B',    # Beginning or one token
            'I',    # Inside (handles MWEs and CHUNKS)
        }
        self.figurative_tags = {'MTP', 'HPB'}  # Remove VIR due to low frequency

    def tokenize(self, text):
        # Use NLTK for Danish tokenization
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

    def create_tag(self, fig_types, prefix):
        if self.tagging_scheme == 'joint':
            return f'{prefix}-{"|".join(sorted(fig_types))}'
        elif self.tagging_scheme == 'metaphor':
            return f'{prefix}-MTP' if 'MTP' in fig_types else 'O'
        elif self.tagging_scheme == 'hyperbole':
            return f'{prefix}-HPB' if 'HPB' in fig_types else 'O'
        elif self.tagging_scheme == 'separate':
            # Returns a tuple of tags (metaphor_tag, hyperbole_tag)
            met_tag = f'{prefix}-MTP' if 'MTP' in fig_types else 'O'
            hyp_tag = f'{prefix}-HPB' if 'HPB' in fig_types else 'O'
            return (met_tag, hyp_tag)

    def tag_document(self, text: str, lexical_units: List[LexicalUnit]):
        tokens = self.tokenize(text)
        if self.tagging_scheme == 'separate':
            tagged_tokens_met = ['O'] * len(tokens)
            tagged_tokens_hyp = ['O'] * len(tokens)
        else:
            tagged_tokens = ['O'] * len(tokens)
        
        span_units = {}
        for unit in lexical_units:
            unit = LexicalUnit(unit[0], unit[1], unit[2], unit[3])
            if unit.fig_type not in self.figurative_tags:
                continue
                
            span_key = tuple(sorted((span.start, span.end) for span in unit.spans))
            if span_key not in span_units:
                span_units[span_key] = []
            span_units[span_key].append(unit)
        
        for spans, units in span_units.items():
            fig_types = set(unit.fig_type for unit in units)
            token_spans = self.map_char_to_token_spans(text, tokens, units[0].spans)
            
            for i, token_index in enumerate(token_spans):
                prefix = 'B' if i == 0 or units[0].unit_type == 'single_word' else 'I'
                if self.tagging_scheme == 'separate':
                    met_tag, hyp_tag = self.create_tag(fig_types, prefix)
                    tagged_tokens_met[token_index] = met_tag
                    tagged_tokens_hyp[token_index] = hyp_tag
                else:
                    tagged_tokens[token_index] = self.create_tag(fig_types, prefix)
        
        if self.tagging_scheme == 'separate':
            return {
                'metaphor': list(zip(tokens, tagged_tokens_met)),
                'hyperbole': list(zip(tokens, tagged_tokens_hyp))
            }
        else:
            return list(zip(tokens, tagged_tokens))

# Apply tagger to all documents in each corpus
def process_corpus(corpus_path, tagging_scheme='joint'):

    all_tagged_documents = {}
    brat = ExtendedBratParser(input_dir=corpus_path, error="ignore")
    examples = list(brat.parse(corpus_path))
    texts = {example.id: example.text for example in examples}
    all_lus = {
        example.id: [(lu.mention, lu.spans, lu.type, lu.tag) for lu in example.lexical_units]
        for example in examples}
        
    tagger = FigurativeLanguageTagger(tagging_scheme=tagging_scheme)
        
    for doc_id, doc_text in texts.items():
        doc_lus = all_lus[doc_id]
        tagged_tokens = tagger.tag_document(doc_text, doc_lus)
        all_tagged_documents[doc_id] = tagged_tokens
            
    return all_tagged_documents

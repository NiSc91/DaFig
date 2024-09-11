import os
import sys
import pandas as pd
import json
from config import *
from pybrat.parser import BratParser, Entity, Event, Example, Relation, Span
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from dataclasses import dataclass

# Define variables
handler = CollectionHandler(ANNOTATED_DIR)
MAIN_PATH = handler.get_collection_path(os.path.join(ANNOTATED_DIR, 'main_collection/main_final'))
OUTPUT_DIR = "temp"

### Create extended classes for parsing the DaFig data ###

class Attribute:
    def __init__(self, id, type, ref_id, value):
        self.id = id
        self.type = type
        self.ref_id = ref_id
        self.value = value

class AttributeParser:
    def __init__(self, input_dir):
        self.input_dir = input_dir

    def parse_attributes(self, doc_id):
        attributes = []
        ann_file = os.path.join(self.input_dir, f"{doc_id}.ann")
        with open(ann_file, "r") as f:
            for line in f:
                if line.startswith("A"):
                    attr_id, info = line.strip().split("\t")
                    attr_type, ref_id, attr_value = info.split()
                    attributes.append(Attribute(attr_id, attr_type, ref_id, attr_value))
        return attributes

class LexicalUnit:
    def __init__(self, text, type, spans, tag, ref_ids):
        self.text = text
        self.type = type
        self.spans = spans
        self.tag = tag
        self.ref_ids = ref_ids

class MWEParser:
    def group_mwe_relations(self, relations, entities):
        mwe_groups = []
        processed = set()

        for relation in relations:
            if relation.type == 'MultiWordExpression':
                arg1, arg2 = relation.arg1, relation.arg2
                
                # Check if either arg1 or arg2 is already in a group
                existing_group = None
                for group in mwe_groups:
                    if arg1 in group or arg2 in group:
                        existing_group = group
                        break

                if existing_group:
                    if arg1 not in existing_group:
                        existing_group.append(arg1)
                    if arg2 not in existing_group:
                        existing_group.append(arg2)
                else:
                    mwe_groups.append([arg1, arg2])

        #print(f"Debug: mwe_groups: {mwe_groups}")
        return mwe_groups

    def create_lexical_units(self, example, mwe_groups):
        lexical_units = []

        # Process MWEs
        for group in mwe_groups:
            spans = sorted([span for entity in group for span in entity.spans], key=lambda s: s.start)
            text = ' '.join(example.text[span.start:span.end] for span in spans)
            tag = group[0].type  # Assuming all entities in MWE have the same tag
            ref_ids = [entity.id for entity in group]

            if any(spans[i+1].start - spans[i].end > 1 for i in range(len(spans)-1)):
                type = 'NonContiguousMWE'
            else:
                type = 'ContiguousMWE'

            lexical_units.append(LexicalUnit(text, type, spans, tag, ref_ids))

        # Process single-word entities (not part of MWEs)
        mwe_entity_ids = set(entity.id for group in mwe_groups for entity in group)
        for entity in example.entities:
            if entity.id not in mwe_entity_ids:
                for span in entity.spans:
                    text = example.text[span.start:span.end]
                    lexical_units.append(LexicalUnit(text, 'single_word', [span], entity.type, [entity.id]))

        return lexical_units

class ExtendedExample(Example):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attributes = []
        self.lexical_units = []
        self.chunks = []
        
    def process_mwe(self, mwe_parser):
        mwe_groups = mwe_parser.group_mwe_relations(self.relations, self.entities)
        self.lexical_units = mwe_parser.create_lexical_units(self, mwe_groups)

    # Filter relations such that only CHUNK relations are included
    def filter_relations(self):
        self.chunks = [relation for relation in self.relations if relation.type == 'CHUNK']
     
     # Filter attributes such that only the first attribute of a lexical unit is included if it's an MWE
    def filter_attribute(self):
        self.attributes = [
            attribute
            for attribute in self.attributes
            if attribute.ref_id in [lex_unit.ref_ids[0] for lex_unit in self.lexical_units]
        ]

    def to_dict(self):
        return {
            'doc_id': self.id,
            'text': self.text,
            'annotations': self.lexical_units,
            'relations': [
                {
                    'id': relation.id,
                    'type': relation.type,
                    'arg1': relation.arg1,
                    'arg2': relation.arg2
                }
                for relation in self.relations if relation.type == 'CHUNK'
            ],
            'attributes': [
                {
                    'id': attribute.id,
                    'type': attribute.type,
                    'ref_id': attribute.ref_id,
                    'value': attribute.value
                }
                for attribute in self.attributes
            ]
        }

class ExtendedBratParser(BratParser):
    def __init__(self, input_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dir = input_dir
        self.attribute_parser = AttributeParser(input_dir)
        self.mwe_parser = MWEParser()

    def parse(self, *args, **kwargs):
        examples = super().parse(*args, **kwargs)
        for example in examples:
            extended_example = ExtendedExample(
                id=example.id,
                text=example.text,
                entities=example.entities,
                relations=example.relations,
                #events=example.events
            )
            extended_example.attributes = self.attribute_parser.parse_attributes(example.id)
            extended_example.process_mwe(self.mwe_parser)
            extended_example.filter_relations()
            yield extended_example

### Parse DaFig data ###

brat = ExtendedBratParser(input_dir=MAIN_PATH, error="ignore")
examples = brat.parse(MAIN_PATH)

### Gather advanced stats ###

# Get lexical units info:
lexical_units = [lu for example in examples for lu in example.lexical_units]
lu_counts = len(lexical_units)

# Get tag info
lu_tag_counts = Counter(lu.tag for lu in lexical_units)

# Get MWE info:
mwe_counts = sum(len(lu.spans) > 1 for lu in lexical_units)
mwe_type_counts = Counter(lu.type for lu in lexical_units)

# Get the 10 most common lexical units tagged with MTP (use counter to get the most common elements)
mtp_lus = [lu for lu in lexical_units if lu.tag == 'MTP']
mtp_lu_counts = Counter(lu.text for lu in mtp_lus)
most_common_mtp_lus = mtp_lu_counts.most_common(10)

### Output results ###

print(f"Total lexical units: {lu_counts}")
print(f"Tag counts: {lu_tag_counts}")
print(f"MWE type counts: {mwe_type_counts}")
print(f"10 most common MTP lexical units: {most_common_mtp_lus}")

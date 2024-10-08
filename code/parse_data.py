import os
import sys
from config import *
from pybrat.parser import BratParser, Entity, Event, Example, Relation, Span
from matplotlib import pyplot as plt

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
    def __init__(self, mention, type, spans, tag, ref_ids):
        self.mention = mention
        self.type = type
        self.spans = spans
        self.tag = tag
        self.ref_ids = ref_ids

class MWEParser:
    def group_relations(self, relations, entities):
        mwe_groups = []
        chunk_groups = []

        for relation in relations:
            if relation.type == 'MultiWordExpression':
                self.add_to_group(mwe_groups, relation.arg1, relation.arg2)
            elif relation.type == 'CHUNK':
                self.add_to_group(chunk_groups, relation.arg1, relation.arg2)

        # Merge overlapping chunks
        self.merge_overlapping_groups(chunk_groups)

        return mwe_groups, chunk_groups

    def add_to_group(self, groups, arg1, arg2):
        existing_group = next((group for group in groups if arg1 in group or arg2 in group), None)
        if existing_group:
            if arg1 not in existing_group:
                existing_group.append(arg1)
            if arg2 not in existing_group:
                existing_group.append(arg2)
        else:
            groups.append([arg1, arg2])

    def merge_overlapping_groups(self, groups):
        i = 0
        while i < len(groups):
            j = i + 1
            while j < len(groups):
                if self.groups_overlap(groups[i], groups[j]):
                    groups[i].extend(groups[j])
                    groups[i] = list(set(groups[i]))  # Remove duplicates
                    groups.pop(j)
                else:
                    j += 1
            i += 1

    def groups_overlap(self, group1, group2):
        spans1 = set(span for entity in group1 for span in entity.spans)
        spans2 = set(span for entity in group2 for span in entity.spans)
        return bool(spans1.intersection(spans2))

    def create_lexical_units(self, example, mwe_groups, chunk_groups):
        lexical_units = []

        # Process MWEs
        for group in mwe_groups:
            spans = sorted([span for entity in group for span in entity.spans], key=lambda s: s.start)
            mention = ' '.join(example.text[span.start:span.end] for span in spans)
            tag = group[0].type  # Assuming all entities in MWE have the same tag
            ref_ids = [entity.id for entity in group]

            if any(spans[i+1].start - spans[i].end > 1 for i in range(len(spans)-1)):
                type = 'NonContiguousMWE'
            else:
                type = 'ContiguousMWE'

            lexical_units.append(LexicalUnit(mention, type, spans, tag, ref_ids))

        # Process CHUNKs
        for group in chunk_groups:
            spans = sorted([span for entity in group for span in entity.spans], key=lambda s: s.start)
            
            # Merge spans if there are gaps
            merged_spans = []
            current_span = spans[0]
            for i in range(1, len(spans)):
                if spans[i].start == current_span.end:
                    current_span = Span(current_span.start, spans[i].end)
                else:
                    merged_spans.append(current_span)
                    current_span = spans[i]
            merged_spans.append(current_span)

            # Create a single span from the first to the last
            full_span = Span(merged_spans[0].start, merged_spans[-1].end)
    
            mention = example.text[full_span.start:full_span.end]
            tag = group[0].type  # Assuming all entities in CHUNK have the same tag
            ref_ids = [entity.id for entity in group]
            lexical_units.append(LexicalUnit(mention, 'CHUNK', [full_span], tag, ref_ids))

        # Process single-word entities (not part of MWEs or CHUNKs)
        grouped_entity_ids = set(entity.id for groups in (mwe_groups, chunk_groups) for group in groups for entity in group)
        for entity in example.entities:
            if entity.id not in grouped_entity_ids:
                lexical_units.append(LexicalUnit(entity.mention, 'single_word', entity.spans, entity.type, [entity.id]))

        return lexical_units

class ExtendedExample(Example):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attributes = []
        self.lexical_units = []
        
    def process_mwe(self, mwe_parser):
        mwe_groups, chunk_groups = mwe_parser.group_relations(self.relations, self.entities)
        self.lexical_units = mwe_parser.create_lexical_units(self, mwe_groups, chunk_groups)

    # Filter attributes such that only the first attribute of a lexical unit is included if it's an MWE or chunk
    def filter_attributes(self):
        self.attributes = [
            attribute
            for attribute in self.attributes
            if attribute.ref_id in [lex_unit.ref_ids[0] for lex_unit in self.lexical_units]
        ]

class ExtendedBratParser(BratParser):
    def __init__(self, input_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dir = input_dir
        self.attribute_parser = AttributeParser(input_dir)
        self.mwe_parser = MWEParser()

    def parse(self, *args, process_chunks=False, **kwargs):
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
            extended_example.filter_attributes()
            yield extended_example

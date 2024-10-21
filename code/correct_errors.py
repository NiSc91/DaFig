from config import *
import json
import os
import logging
from typing import List, Tuple, Dict
import peek

def process_corpus(corpus_path: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Process the corpus at the given path, ask for confirmation before removing empty documents,
    and log the results.

    Args:
    corpus_path (str): Path to the corpus directory

    Returns:
    Tuple[List[str], List[str], List[str]]: Lists of removed, kept, and failed to remove empty documents
    """
    # Set up logging
    logging.basicConfig(filename=os.path.join(TEMP_DIR, 'corpus_processing.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Read in the corpus
    corpus = peek.AnnCorpus(corpus_path, txt=True)

    # Get a list of empty (un-annotated) documents
    empty_docs = corpus.get_empty_files()

    logging.info(f"Found {len(empty_docs)} empty documents in {corpus_path}")

    removed_docs = []
    kept_docs = []
    failed_removals = []

    # Process empty documents
    for doc_path in empty_docs:
        print(f"\nEmpty document found: {doc_path}")
        choice = input("Do you want to remove this document? (y/n): ").lower().strip()
        
        if choice == 'y':
            try:
                # Remove .ann file
                os.remove(doc_path)
                # Remove corresponding .txt file
                txt_path = doc_path[:-3] + 'txt'
                os.remove(txt_path)
                removed_docs.append(doc_path)
                logging.info(f"User confirmed removal of empty document: {doc_path}")
            except Exception as e:
                failed_removals.append(doc_path)
                logging.error(f"Failed to remove empty document {doc_path}: {str(e)}")
        else:
            kept_docs.append(doc_path)
            logging.info(f"User chose to keep empty document: {doc_path}")

    logging.info(f"Removed {len(removed_docs)} empty documents")
    logging.info(f"Kept {len(kept_docs)} empty documents")
    logging.info(f"Failed to remove {len(failed_removals)} empty documents")

    return removed_docs, kept_docs, failed_removals

### Correct misaligned annotations ###
def load_corrections(json_file: str) -> Dict[str, List[Dict]]:
    corrections = {}
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            correction = json.loads(line)
            doc_id = correction['doc_ID']
            if doc_id not in corrections:
                corrections[doc_id] = []
            corrections[doc_id].append(correction)
    return corrections

def apply_corrections(ann_file: str, doc_corrections: List[Dict]) -> Tuple[List[str], int]:
    with open(ann_file, 'r') as f:
        lines = f.readlines()

    changes_made = 0
    new_lines = []

    for line in lines:
        if line.startswith('T'):  # Entity line
            entity_id = line.split('\t')[0]
            correction = next((c for c in doc_corrections if c['entity_ID'] == entity_id), None)
            
            if correction:
                parts = line.split('\t')
                tag = parts[1].split()[0]
                old_start, old_end = map(int, parts[1].split()[1:3])
                old_mention = parts[2].strip()
                new_start = correction['suggested_span']['start']
                new_end = correction['suggested_span']['end']
                new_mention = correction['suggested_mention']
                
                print(f"\nFile: {ann_file}")
                print(f"Entity ID: {entity_id}")
                print(f"Current: {old_start} {old_end}\t{old_mention}")
                print(f"Suggested: {new_start} {new_end}\t{new_mention}")
                
                while True:
                    choice = input("Accept this change? (y/n): ").lower()
                    if choice in ['y', 'n']:
                        break
                    print("Please enter 'y' for yes or 'n' for no.")
                
                if choice == 'y':
                    new_line = f"{entity_id}\t{tag} {new_start} {new_end}\t{new_mention}\n"
                    new_lines.append(new_line)
                    changes_made += 1
                    logging.info(f"Change accepted in {ann_file} for entity {entity_id}")
                else:
                    new_lines.append(line)
                    logging.info(f"Change rejected in {ann_file} for entity {entity_id}")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return new_lines, changes_made

def process_files(corrections: Dict[str, List[Dict]], ann_dir: str) -> Tuple[int, int]:
    total_files_processed = 0
    total_changes_made = 0

    for doc_id, doc_corrections in corrections.items():
        ann_file = os.path.join(ann_dir, f"{doc_id}.ann")
        if os.path.exists(ann_file):
            updated_lines, changes_made = apply_corrections(ann_file, doc_corrections)
            with open(ann_file, 'w') as f:
                f.writelines(updated_lines)
            total_files_processed += 1
            total_changes_made += changes_made
            logging.info(f"Processed {ann_file} with {changes_made} changes")
        else:
            logging.warning(f"File not found: {ann_file}")

    return total_files_processed, total_changes_made

def process_single_corpus(corpus_path: str, corrections_file: str, output_dir: str) -> None:
    # Set up logging for this corpus
    log_file = os.path.join(output_dir, f"{os.path.basename(corpus_path)}_processing.log")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Processing corpus: {corpus_path}")

    # Process the corpus (remove empty documents)
    removed, kept, failed = process_corpus(corpus_path)

    # Print and log summary of empty document removal
    summary = f"\nEmpty Document Removal Summary for {corpus_path}:\n"
    summary += f"Removed {len(removed)} empty documents\n"
    summary += f"Kept {len(kept)} empty documents\n"
    summary += f"Failed to remove {len(failed)} empty documents\n"
    print(summary)
    logging.info(summary)

    # Process corrections
    corrections = load_corrections(corrections_file)
    files_processed, changes_made = process_files(corrections, corpus_path)

    # Print and log summary of corrections
    summary = f"\nCorrections Summary for {corpus_path}:\n"
    summary += f"Processed {files_processed} files\n"
    summary += f"Made {changes_made} changes\n"
    print(summary)
    logging.info(summary)

    logging.info(f"Corpus processing completed for {corpus_path}.")

def main():
    # Declare variables
    handler = CollectionHandler(CORPORA_DIR)

    all_corpora = handler.get_collections()
    print("All annotated corpora:", all_corpora)

    corpora_paths = {
        #'main': handler.get_collection_path(os.path.join(CORPORA_DIR, 'main')),
        #'agr1': handler.get_collection_path(os.path.join(CORPORA_DIR, 'agr1')),
        #'agr2': handler.get_collection_path(os.path.join(CORPORA_DIR, 'agr2')),
        #'agr3': handler.get_collection_path(os.path.join(CORPORA_DIR, 'agr3')),
        #'agr_final': handler.get_collection_path(os.path.join(CORPORA_DIR, 'agr_final')),
        #'consensus_agr2': handler.get_collection_path(os.path.join(CORPORA_DIR, 'consensus_agr2')),
    }

    OUTPUT_DIR = TEMP_DIR

    # Process each corpus
    for corpus_name, corpus_path in corpora_paths.items():
        CORRECTIONS_FILE = os.path.join(OUTPUT_DIR, f"misaligned_spans_{corpus_name}.json")
        print(f"\nProcessing corpus: {corpus_name}")
        process_single_corpus(corpus_path, CORRECTIONS_FILE, OUTPUT_DIR)

    print("\nAll corpora have been processed.")

if __name__ == "__main__":
    main()
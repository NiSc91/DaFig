import os
from config import *
from parse_data import ExtendedBratParser

# Variables
handler = CollectionHandler(CORPORA_DIR)
all_corpora = handler.get_collections()
print("All annotated corpora:", all_corpora)

# Create paths for the different corpora
CORPUS_NAMES = ['main', 'agr1', 'agr2', 'agr3', 'agr_combined', 'consensus', 'reanno']
CORPUS_PATHS = {f"{name.upper()}_PATH": handler.get_collection_path(os.path.join(CORPORA_DIR, name)) for name in CORPUS_NAMES}
OUTPUT_DIR = RESULTS_DIR

# Check if the output directory exists, if not, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Create a lambda function to get ann paths
get_ann_path = lambda base_path, ann_folder: os.path.join(base_path, ann_folder)

# Create a dictionary with the paths to the ann folders for each corpus except for main
base_paths = {name: handler.get_collection_path(os.path.join(CORPORA_DIR, name)) for name in CORPUS_NAMES[1:]}
ann_paths = {f"{name.upper()}_ANN1_PATH": get_ann_path(base_path, 'ann1') for name, base_path in base_paths.items()}
ann_paths.update({f"{name.upper()}_ANN2_PATH": get_ann_path(base_path, 'ann2') for name, base_path in base_paths.items()})

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def validate_ann_files(corpus_path):
    brat = ExtendedBratParser(input_dir=corpus_path, error="ignore")
    successful_files = []
    error_files = []

    for root, dirs, files in os.walk(corpus_path):
        for file in files:
            if file.endswith('.ann'):
                full_path = os.path.join(root, file)
                logger.info(f"Processing file: {full_path}")
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    try:
                        examples = list(brat.parse(full_path))
                        logger.debug(f"Parsed examples: {examples}")
                        successful_files.append(full_path)
                        logger.info(f"Successfully parsed file: {full_path}")
                        
                        # Print out some information about the parsed content
                        print(f"\nFile: {full_path}")
                        print(f"Number of parsed items: {len(examples)}")
                        for i, example in enumerate(examples[:5]):  # Print details of first 5 items
                            print(f"Item {i+1}: {example}")
                        if len(examples) > 5:
                            print("...")
                        
                    except Exception as parse_error:
                        error_files.append((full_path, str(parse_error)))
                        logger.error(f"Error parsing file {full_path}: {str(parse_error)}")
                        print(f"\nError parsing file {full_path}:")
                        print(f"Error: {str(parse_error)}")
                        print("First few lines of the file:")
                        for i, line in enumerate(lines[:5]):
                            print(f"{i+1}: {line.strip()}")
                except Exception as e:
                    error_files.append((full_path, str(e)))
                    logger.error(f"Error processing file {full_path}: {str(e)}")
                    print(f"\nError processing file {full_path}:")
                    print(f"Error: {str(e)}")
                    print("First few lines of the file:")
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            for i, line in enumerate(f):
                                if i < 5:
                                    print(f"{i+1}: {line.strip()}")
                                else:
                                    break
                    except Exception as read_error:
                        logger.error(f"Error reading file {full_path}: {str(read_error)}")
                        print(f"Error reading file: {str(read_error)}")

    print("\nSuccessfully parsed files:")
    for file in successful_files:
        print(file)

    print("\nFiles with errors:")
    for file, error in error_files:
        print(f"{file}: {error}")

    print(f"\nTotal files: {len(successful_files) + len(error_files)}")
    print(f"Successfully parsed: {len(successful_files)}")
    print(f"Errors: {len(error_files)}")

# Use the path to your 'reanno' corpus
validate_ann_files(CORPUS_PATHS['REANNO_PATH'])

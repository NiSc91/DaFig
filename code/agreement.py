import os
from config import *
import subprocess
import peek

## Declare variables
handler = CollectionHandler(CORPORA_DIR)
AGR_NAMES = ['agr1', 'agr2', 'agr3', 'agr_combined', 'consensus']
AGR_PATHS = {f"{name.upper()}_PATH": handler.get_collection_path(os.path.join(CORPORA_DIR, name)) for name in AGR_NAMES}
OUTPUT_DIR = TEMP_DIR

# Create a lambda function to get ann paths
get_ann_path = lambda base_path, ann_folder: os.path.join(base_path, ann_folder)
ann_paths = {f"{name.upper()}_ANN1_PATH": get_ann_path(base_path, 'ann1') for name, base_path in AGR_PATHS.items()}
ann_paths.update({f"{name.upper()}_ANN2_PATH": get_ann_path(base_path, 'ann2') for name, base_path in AGR_PATHS.items()})

### Helper functions ###
def extract_tar_gz_file(file_path, destination_path):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=destination_path)

def rename_files(directory):
    mappings = []

    for filename in os.listdir(directory):
        if "_" in filename:
            prefix, suffix = filename.split("_", 1)
            new_name = suffix
            mappings.append({prefix: new_name})
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
    
    return mappings

## Create brat_peek corpus objects for the AGR_COMBINED collection
ann_corpus_ann1 = peek.AnnCorpus(ann_paths['AGR3_PATH_ANN1_PATH'])
ann_corpus_ann2 = peek.AnnCorpus(ann_paths['AGR3_PATH_ANN2_PATH'])

# Calculate agreement
peek.metrics.show_iaa([ann_corpus_ann1, ann_corpus_ann2], ['filename', 'label', 'offset'], ['MTP', 'HPB', 'WIDLII'], tsv=True)

def run_brateval(evaluation_folder, groundtruth_folder, span_match='exact', type_match='exact', threshold=None, verbose=False, full_taxonomy=False, config_path=None):
    command = [
        'java',
        '-cp',
        JAR_PATH,
        'au.com.nicta.csp.brateval.CompareEntities',
        evaluation_folder,
        groundtruth_folder
    ]
    
    # Handle span matching
    if span_match.lower() != 'exact':
        command.extend(['-s', span_match])
        if span_match.lower() == 'approx' and threshold is not None:
            command.append(str(threshold))
    
    # Handle type matching
    if type_match.lower() != 'exact':
        command.extend(['-t', type_match])
    
    # Add verbose option if requested
    if verbose:
        command.append('-v')
    
    # Add full taxonomy option if requested
    if full_taxonomy:
        command.append('-ft')
    
    # Add config file path if provided
    if config_path:
        command.extend(['-config', config_path])
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running BratEval: {e}")
        print(f"Error output: {e.stderr}")
        return None

# Run brateval for AGR_COMBINED path with exact span and type match
output = run_brateval(
    ann_paths['AGR3_PATH_ANN1_PATH'],
    ann_paths['AGR3_PATH_ANN2_PATH'],
    verbose=False,
    full_taxonomy=True,
    config_path=CORPORA_DIR
)   
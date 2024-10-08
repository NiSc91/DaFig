import os
from config import *
import subprocess

# Get agreement collection
CORPUS_PATHS = {
    'samuel_combined': os.path.join(ANNOTATED_DIR, "agr_collection/combined_agr/Samuel"),
    'stephanie_combined': os.path.join(ANNOTATED_DIR, "agr_collection/combined_agr/Stephanie"),
# Uncomment and add other corpus paths as needed
 'samuel_sample3': os.path.join(ANNOTATED_DIR, 'agr_collection/Samuel/sample3'),
 'stephanie_sample3': os.path.join(ANNOTATED_DIR, 'agr_collection/Stephanie/sample3'),
# ... and so on
}

## Helper functions
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

##Define annotators and create brat_peek corpus objects
import peek

ANNOTATOR_1 = "Samuel"
ANNOTATOR_2 = "Stephanie"

ann_corpus_ann1 = peek.AnnCorpus(CORPUS_PATHS[f'{ANNOTATOR_1.lower()}_combined'])
ann_corpus_ann2 = peek.AnnCorpus(CORPUS_PATHS[f'{ANNOTATOR_2.lower()}_combined'])

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

output = run_brateval(
    CORPUS_PATHS['samuel_combined'],
    CORPUS_PATHS['stephanie_combined'],
    span_match='approx',
    threshold=0.8,
    type_match='exact',
    verbose=False,
    full_taxonomy=True,
    config_path=ANNOTATED_DIR
)   
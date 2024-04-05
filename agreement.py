import os
import tarfile
import sys

if not os.path.exists('temp'):
    os.makedirs('temp')

def extract_tar_gz_file(file_path, destination_path):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=destination_path)

sys.path.append('brat_peek')

data_path = "brat_data/annotated"
extract_tar_gz_file(os.path.join(data_path, 'agr_collection.tar.gz'), data_path)

corpus1a = os.path.join(data_path, 'agr_collection/Samuel/sample1')
corpus1b = os.path.join(data_path, 'agr_collection/Stephanie/sample1')
corpus2a = os.path.join(data_path, 'agr_collection/Samuel/sample2')
corpus2b = os.path.join(data_path, 'agr_collection/Stephanie/sample2')

def rename_files(directory):
    mappings = []

    for filename in os.listdir(directory):
        if "_" in filename:
            prefix, suffix = filename.split("_", 1)
            new_name = suffix
            mappings.append({prefix: new_name})
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
    
    return mappings

# Rename files in both directories and get the mappings
#mappings_corpus1 = rename_files(corpus1a)
#mappings_corpus2 = rename_files(corpus1b)

from brat_peek import peek

ann_corpus1a = peek.AnnCorpus(corpus1a)
ann_corpus1b = peek.AnnCorpus(corpus1b)
ann_corpus2a = peek.AnnCorpus(corpus2a)
ann_corpus2b = peek.AnnCorpus(corpus2b)

# Calculate agreement
#peek.metrics.show_iaa([ann_corpus1a, ann_corpus1b], ['filename', 'label', 'offset'], ann_corpus1a.text_labels, tsv=True)
# Calculate agreement
peek.metrics.show_iaa([ann_corpus2a, ann_corpus2b], ['filename', 'label', 'offset'], ann_corpus2a.text_labels, tsv=True)

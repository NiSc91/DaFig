import pdb
import os
import tarfile
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)  # Add the parent folder to the module search path
sys.path.append(parent_dir+"/brat_peek")  # Add the parent folder to the module search path

if not os.path.exists('temp'):
    os.makedirs('temp')

def extract_tar_gz_file(file_path, destination_path):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=destination_path)

data_path = "brat_data/annotated"
#extract_tar_gz_file(os.path.join(data_path, 'agr_collection.tar.gz'), data_path)

corpus1a = os.path.join(data_path, 'agr_collection/Samuel/sample1')
corpus1b = os.path.join(data_path, 'agr_collection/Stephanie/sample1')
corpus2a = os.path.join(data_path, 'agr_collection/Samuel/sample2')
corpus2b = os.path.join(data_path, 'agr_collection/Stephanie/sample2')
corpus3a = os.path.join(data_path, 'agr_collection/Samuel/sample3')
corpus3b = os.path.join(data_path, 'agr_collection/Stephanie/sample3')
corpus1_combined = os.path.join(data_path, "agr_collection/combined_agr/Samuel")
corpus2_combined = os.path.join(data_path, "agr_collection/combined_agr/Stephanie")

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
mappings_corpus1 = rename_files(corpus1_combined)
mappings_corpus2 = rename_files(corpus2_combined)

from brat_peek import peek

#ann_corpus1a = peek.AnnCorpus(corpus1a)
#ann_corpus1b = peek.AnnCorpus(corpus1b)
#ann_corpus2a = peek.AnnCorpus(corpus2a)
#ann_corpus2b = peek.AnnCorpus(corpus2b)
ann_corpus3a = peek.AnnCorpus(corpus3a)
ann_corpus3b = peek.AnnCorpus(corpus3b)
ann_corpus1_combined = peek.AnnCorpus(corpus1_combined)
ann_corpus2_combined = peek.AnnCorpus(corpus2_combined)

## Find all annotations with 'WIDLII' in ann_corpus3a
#docs = [doc for doc in ann_corpus3a.docs if 'WIDLII' in doc.count['entities']]

# Calculate agreement
#peek.metrics.show_iaa([ann_corpus1a, ann_corpus1b], ['filename', 'label', 'offset'], ann_corpus1a.text_labels, tsv=True)
#peek.metrics.show_iaa([ann_corpus2a, ann_corpus2b], ['filename', 'label', 'offset'], ann_corpus2a.text_labels, tsv=True)
peek.metrics.show_iaa([ann_corpus3a, ann_corpus3b], ['filename', 'label', 'offset'], ann_corpus3b.text_labels, tsv=True) # Stephanie (Corpus3b) doesn't use the WIDLII label at all
#peek.metrics_mod.show_iaa([ann_corpus1_combined, ann_corpus2_combined], ['filename', 'label', 'offset'], ann_corpus1_combined.text_labels, tsv=False)

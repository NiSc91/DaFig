import os
import shutil
from config import *
import peek

# Get the specific collections
handler = CollectionHandler(ANNOTATED_DIR)

all_corpora = handler.get_collections()
print("All annotated corpora:", all_corpora)

main_path = handler.get_collection_path("main_collection")
agr_path = handler.get_collection_path("agr_collection")
subcollections = handler.get_subcollections("agr_collection")

## Read in the corpus and perform some basic operations
main_corpus = peek.AnnCorpus(main_path, txt=True)
doc = main_corpus.get_random_doc()
print("Random corpus document from the main collection:")
print()
print(doc.anns.items())

## Get a list of empty (un-annotated) documents and move non-empty documents to a new subcollection
empty_docs = main_corpus.get_empty_files()
annotated_docs = [doc.path for doc in main_corpus.docs if doc.path not in empty_docs]

## Create new subcollection called 'main_final'
new_subcollection_path = os.path.join(main_path, "main_final")
os.makedirs(new_subcollection_path, exist_ok=True)

# Copy non-empty documents (meaning the ann-files in annotated_docs and corresponding txt-files) to the new subcollection
for ann_path in annotated_docs:
    # Get the corresponding txt-file from the ann-file (all ann_paths ends with .ann)
    txt_path = ann_path[:-3] + 'txt'
    # Copy the ann-file and txt-file to the new subcollection in case they don't already exist
    if not os.path.exists(os.path.join(new_subcollection_path, os.path.basename(ann_path))):
        shutil.copy(ann_path, new_subcollection_path)
    if not os.path.exists(os.path.join(new_subcollection_path, os.path.basename(txt_path))):
        shutil.copy(txt_path, new_subcollection_path)


### Rudimentary analysis using brat_peek library ###

def analyze_corpus(input_dir, output_dir):
    # Initialize corpus
    corpus = peek.AnnCorpus(input_dir, txt=True)

    # Generate basic statistics
    print('Corpus stats:', corpus.count)
    print('Entity labels found in corpus:', corpus.text_labels)
    print('Document 42 annotations:', corpus.docs[41].anns)

    # Generate and save plot
    peek.stats.plot_tags(corpus, save_fig=True, outpath=os.path.join(output_dir, 'corpus_stats.png'))

    # Generate .tsv with statistics
    peek.stats.generate_corpus_stats_tsv(corpus, include_txt=True, out_path=output_dir)

    # Filter text frequency to exclude stop words
    stop_words = set(stopwords.words('danish'))
    filtered_text_freq = {
        entity: Counter({word: freq for word, freq in text_freq.items() if word not in stop_words})
        for entity, text_freq in corpus.text_freq.items()
    }

    # Get the 10 most common metaphors and hyperboles
    print("10 most common metaphors:", filtered_text_freq['MTP'].most_common(10))
    print("10 most common hyperboles:", filtered_text_freq['HPB'].most_common(10))

    return corpus, filtered_text_freq

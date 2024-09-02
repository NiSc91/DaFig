import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = "brat_data/annotated"

sys.path.append(parent_dir)  # Add the parent folder to the module search path
sys.path.append(parent_dir+"/brat_peek")  # Add the parent folder to the module search path

from brat_peek import peek

class CollectionHandler:
    def __init__(self, data_path):
        self.data_path = data_path

    def get_collections(self):
        collections = []
        for root, dirs, files in os.walk(self.data_path):
            for dir_name in dirs:
                collections.append(dir_name)
        return collections

    def get_collection_path(self, collection_name):
        collection_path = os.path.join(self.data_path, collection_name)
        if not os.path.exists(collection_path):
            raise ValueError("Collection '{}' does not exist".format(collection_name))
        return collection_path

    def get_subcollections(self, collection_name):
        collection_path = self.get_collection_path(collection_name)
        subcollections = []
        for root, dirs, files in os.walk(collection_path):
            for dir_name in dirs:
                subcollections.append(dir_name)
        return subcollections

    def extract_ids_and_collections(self, target_collection=None):
        ids_and_collections = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    collection = self.get_collection_from_path(file_path)
                    if collection and (collection == target_collection or target_collection is None):
                        file_id = os.path.splitext(file)[0]
                        ids_and_collections.append((file_id, collection))
        return ids_and_collections

    def get_collection_from_path(self, file_path):
        collection_path = os.path.dirname(file_path)
        collection = os.path.basename(collection_path)
        return collection

# Example usage
data_path = "brat_data/annotated"
handler = CollectionHandler(data_path)

all_corpora = handler.get_collections()
print("All annotated corpora:", all_corpora)

main_path = handler.get_collection_path("main_collection")
agr_path = handler.get_collection_path("agr_collection")
subcollections = handler.get_subcollections("agr_collection")

corpus = peek.AnnCorpus(main_path)
doc = corpus.get_random_doc()
print("Random corpus document from the main collection:")
print()
print(doc.anns.items())

# Statistics
print('Corpus stats:', corpus.count)
print('Entity labels found in corpus: ', corpus.text_labels)
print('Document 42 annotations:', corpus.docs[41].              
      anns)
# Create a plot
peek.stats.plot_tags(corpus)
# Generate .tsv with statistics
peek.stats.generate_corpus_stats_tsv(corpus, include_txt=True, out_path="temp/")
print(corpus.text_freq)
print("Most common metaphors:")
print(corpus.text_freq['MTP'].most_common(5))
print(corpus.text_freq_lower)
print("Most common hyperboles:")
print(corpus.text_freq['HPB'].most_common(5))
print(corpus.text_freq_lower)

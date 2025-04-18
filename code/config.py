import os
import tarfile
import sys

# Define base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
DATA_DIR = os.path.join(PARENT_DIR, 'brat_data')
DEPS_DIR = os.path.join(PARENT_DIR, 'deps')
RESULTS_DIR = os.path.join(PARENT_DIR,'results')
MODELS_DIR = os.path.join(PARENT_DIR,'models')

# Add necessary directories to sys.path
sys.path.extend([
    PARENT_DIR,
    os.path.join(DEPS_DIR, "brat-peek")
])

# Define other important directories
BRATEVAL_DIR = os.path.join(DEPS_DIR, 'brateval')
JAR_PATH = os.path.join(BRATEVAL_DIR, 'target/brateval.jar')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
CORPORA_DIR = os.path.join(DATA_DIR, 'DaFig_data/corpora') # Also where to find backup and annotation.conf file

# Create temp directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)

## Collection handler
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

    def get_txt_files(self, collection_path):
        txt_files = []
        for root, dirs, files in os.walk(collection_path):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(file)
        return txt_files
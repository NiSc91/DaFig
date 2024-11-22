from config import *
from bio_encoder import process_corpus
from train_dafig import *
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Variables
# Variables
handler = CollectionHandler(CORPORA_DIR)
all_corpora = handler.get_collections()
print("All annotated corpora:", all_corpora)

# Create paths for the agreement corpora
CORPUS_NAMES = ['main', 'agr1', 'agr2', 'agr3', 'agr_combined', 'consensus']
CORPUS_PATHS = {f"{name.upper()}_PATH": handler.get_collection_path(os.path.join(CORPORA_DIR, name)) for name in CORPUS_NAMES}
OUTPUT_DIR = os.path.join(CORPORA_DIR, "../BIO")

# Check if the output directory exists, if not, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Create a lambda function to get ann paths
get_ann_path = lambda base_path, ann_folder: os.path.join(base_path, ann_folder)

# Create a dictionary with the paths to the ann folders for each corpus except for main
base_paths = {name: handler.get_collection_path(os.path.join(CORPORA_DIR, name)) for name in CORPUS_NAMES[1:]}
ann_paths = {f"{name.upper()}_ANN1_PATH": get_ann_path(base_path, 'ann1') for name, base_path in base_paths.items()}
ann_paths.update({f"{name.upper()}_ANN2_PATH": get_ann_path(base_path, 'ann2') for name, base_path in base_paths.items()})

### Compare human annotators to each other ###

def calculate_bilateral_f1(ann1_labels, ann2_labels, id2label):
    """
    Calculate macro F1 scores treating each annotator as ground truth in turn.
    
    Args:
        ann1_labels: Labels from first annotator
        ann2_labels: Labels from second annotator
        id2label: Mapping from ID to label names
    
    Returns:
        dict: Dictionary containing F1 scores in both directions and average
    """
    # Calculate metrics treating ann1 as ground truth
    f1_ann1_as_truth = f1_score(ann1_labels, ann2_labels, average='macro')
    
    # Calculate metrics treating ann2 as ground truth
    f1_ann2_as_truth = f1_score(ann2_labels, ann1_labels, average='macro')
    
    # Calculate class-wise F1 scores for detailed analysis
    class_f1_ann1_as_truth = f1_score(ann1_labels, ann2_labels, average=None)
    
    # Print detailed results
    print("\nDetailed F1 scores per class (ann1 as ground truth):")
    for i, score in enumerate(class_f1_ann1_as_truth):
        print(f"Class {id2label[i]}: {score:.4f}")
    
    print(f"\nMacro F1 (ann1 as ground truth): {f1_ann1_as_truth:.4f}")
    print(f"Macro F1 (ann2 as ground truth): {f1_ann2_as_truth:.4f}")
    print(f"Average Macro F1: {(f1_ann1_as_truth + f1_ann2_as_truth) / 2:.4f}")
    
    return {
        'f1_ann1_as_truth': f1_ann1_as_truth,
        'f1_ann2_as_truth': f1_ann2_as_truth,
        'average_f1': (f1_ann1_as_truth + f1_ann2_as_truth) / 2,
        'class_wise_f1': class_f1_ann1_as_truth
    }

def compare_annotations(ann1_path, ann2_path):
    # Prepare data for both annotators
    docs1, labels1, label2id, id2label = prepare_data(ann1_path)
    docs2, labels2, _, _ = prepare_data(ann2_path, label2id=label2id)
    
    # Flatten the labels for comparison
    flat_labels1 = [label for doc_labels in labels1 for label in doc_labels]
    flat_labels2 = [label for doc_labels in labels2 for label in doc_labels]
    
    # Calculate bilateral F1 scores
    f1_metrics = calculate_bilateral_f1(flat_labels1, flat_labels2, id2label)
    
    # Calculate confusion matrix
    cm = confusion_matrix(flat_labels1, flat_labels2)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Print label mapping for confusion matrix interpretation
    print("\nLabel mapping:")
    for i, label in id2label.items():
        print(f"{i}: {label}")
    
    return f1_metrics, cm, id2label

def main():
    # Use the paths from your existing code
    ann1_path = ann_paths['AGR_COMBINED_ANN1_PATH']
    ann2_path = ann_paths['AGR_COMBINED_ANN2_PATH']
    
    print("Comparing annotations between two annotators...")
    metrics, cm, id2label = compare_annotations(ann1_path, ann2_path)
    
if __name__ == "__main__":
    main()


### Compare to model predictions ###
# Load saved model and make predictions


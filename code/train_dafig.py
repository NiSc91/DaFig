from transformers import (
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizerFast,
    LongformerForTokenClassification,
    LongformerTokenizerFast
    )
from config import *
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import math
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
from bio_encoder import process_corpus

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

# Model and training config
def get_model_config(model_type, num_labels):
    base_config = {
        # Training configurations (stays same for all models)
        'batch_size': 16,
        'num_epochs': 20,
        'learning_rate': 2e-5,
        'train_split': 0.8,
        'early_stopping_patience': 3,
        'num_labels': num_labels  # Now directly using num_labels
    }
    
    # Model-specific configurations
    model_configs = {
        'distilbert': {
            'model_type': 'distilbert',
            'model_name': 'distilbert-base-multilingual-cased',
            'max_length': 512,
        },
        'xlmroberta': {
            'model_type': 'xlmroberta',
            'model_name': 'xlm-roberta-base',
            'max_length': 512,
        },
        'longformer': {
            'model_type': 'longformer',
            'model_name': 'allenai/longformer-base-4096',
            'max_length': 4096,
        }
    }
    
    # Combine base config with model-specific config
    config = {**base_config, **model_configs[model_type]}
    return config

### Preprocess the data ###

# Prepare dataset

def prepare_data(corpus_path, tagging_scheme='joint', label2id=None):
    """
    Prepare data for a single corpus.
    
    Args:
        corpus_path (str): Path to the corpus
        tagging_scheme (str, optional): Tagger to use for extracting labels. Defaults to 'joint'
        label2id (dict, optional): Existing label mapping. If None, a new mapping will be created
    
    Returns:
        documents (list): List of tokenized documents
        label_ids (list): List of label IDs
        label2id (dict): Mapping of labels to IDs
        id2label (dict): Mapping of IDs to labels
    """
    # Get tagged documents from single corpus
    data = process_corpus(corpus_path, tagging_scheme=tagging_scheme)
    
    # Process the data
    documents = []
    labels = []
    
    for doc_id, tokens in data.items():
        doc_tokens = []
        doc_labels = []
        for token, label in tokens:
            doc_tokens.append(token)
            doc_labels.append(label)
        documents.append(doc_tokens)
        labels.append(doc_labels)
    
    # Create or update label mappings
    if label2id is None:
        unique_labels = set([label for doc_labels in labels for label in doc_labels])
        label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    # Convert labels to IDs
    label_ids = [[label2id[label] for label in doc_labels] for doc_labels in labels]
    
    # Print some basic statistics
    print(f"\nCorpus Statistics for {corpus_path}:")
    print(f"Number of documents: {len(documents)}")
    print(f"Number of unique labels in this corpus: {len(unique_labels)}")

    return documents, label_ids, label2id, id2label

# Split sequences longer than max_len
def split_long_document(text, labels, max_len):
    # Create overlapping chunks for long documents
    chunks_text = []
    chunks_labels = []
    stride = max_len // 2  # 50% overlap
    
    for i in range(0, len(text), stride):
        chunk_text = text[i:i + max_len]
        chunk_labels = labels[i:i + max_len]
        if len(chunk_text) > max_len // 4:  # Only keep chunks with substantial content
            chunks_text.append(chunk_text)
            chunks_labels.append(chunk_labels)
    
    return chunks_text, chunks_labels

# Analyze sequence length before training
def analyze_sequence_lengths(dataset):
    lengths = [len(text) for text in dataset.processed_texts]
    print(f"Max length: {max(lengths)}")
    print(f"Mean length: {sum(lengths)/len(lengths):.2f}")
    print(f"Number of splits: {len(dataset.processed_texts)}")

### Dataset and data loader ###
# Custom dataset and data loader
class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):  # Changed from init to __init__
        self.processed_texts = []
        self.processed_labels = []
        self.chunk_to_doc_mapping = []
        
        doc_idx = 0
        valid_labels_count = 0
        total_labels_count = 0
        
        for text, label in zip(texts, labels):
            if len(text) > max_len:
                chunk_texts, chunk_labels = split_long_document(text, label, max_len)
                self.processed_texts.extend(chunk_texts)
                self.processed_labels.extend(chunk_labels)
                for _ in range(len(chunk_texts)):
                    self.chunk_to_doc_mapping.append(doc_idx)
            else:
                self.processed_texts.append(text)
                self.processed_labels.append(label)
                self.chunk_to_doc_mapping.append(doc_idx)
            
            # Count valid labels
            valid_labels_count += sum(1 for l in label if l != -100)
            total_labels_count += len(label)
            
            doc_idx += 1
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        print(f"\nDataset Statistics:")
        print(f"Total documents: {doc_idx}")
        print(f"Total chunks: {len(self.processed_texts)}")
        print(f"Valid labels: {valid_labels_count}/{total_labels_count} ({valid_labels_count/total_labels_count*100:.2f}%)")
        
    def __getitem__(self, idx):  # Changed from getitem to __getitem__
        text = self.processed_texts[idx]
        label = self.processed_labels[idx]

        encoding = self.tokenizer(text, 
                                is_split_into_words=True, 
                                max_length=self.max_len, 
                                truncation=True, 
                                padding='max_length')

        word_ids = encoding.word_ids()
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(label_ids)
        }

    def __len__(self):  # Added __len__ method
        return len(self.processed_texts)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Modified training loop with early stopping and reduced learning rate
def train_model(model, train_dataloader, val_dataloader, num_epochs, device, 
                initial_lr=2e-5, patience=3):
    
    # Calculate class weights from training data
    train_labels = []
    for batch in train_dataloader:
        labels = batch['labels']
        train_labels.extend(labels[labels != -100].cpu().numpy())

    # Compute class weights
    unique_classes = np.unique(train_labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=train_labels
    )
    class_weights = torch.FloatTensor(class_weights).to(device)

    # Define criterion with weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=initial_lr, 
                                weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        verbose=True
    )
    
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits.view(-1, len(unique_classes)), labels.view(-1))
            
            total_train_loss += loss.item()
            
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits.view(-1, len(unique_classes)), labels.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average validation loss: {avg_val_loss:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print("New best model saved!")
        
        # Early stopping check
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

# evaluation metrics for split-aware learning
def evaluate_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    overlap_predictions = []
    overlap_labels = []
    
    max_len = 512
    overlap_size = max_len // 2
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            for pred, label, mask in zip(predictions, labels, attention_mask):
                valid_mask = (label != -100) & (mask == 1)
                valid_pred = pred[valid_mask].cpu().numpy()
                valid_label = label[valid_mask].cpu().numpy()
                
                if len(valid_pred) >= overlap_size:
                    content_length = len(valid_pred)
                    start_idx = content_length // 4
                    end_idx = 3 * content_length // 4
                    
                    overlap_region_pred = valid_pred[start_idx:end_idx]
                    overlap_region_label = valid_label[start_idx:end_idx]
                    
                    overlap_predictions.extend(overlap_region_pred)
                    overlap_labels.extend(overlap_region_label)
                
                all_predictions.extend(valid_pred)
                all_labels.extend(valid_label)

    # Calculate metrics for both minority and majority classes
    metrics = {}
    
    # Overall metrics
    metrics.update({
        'overall_f1_weighted': f1_score(all_labels, all_predictions, 
                                    average='weighted', zero_division=0),
        'overall_f1_macro': f1_score(all_labels, all_predictions, 
                                average='macro', zero_division=0),
        'overall_precision': precision_score(all_labels, all_predictions, 
                                        average='weighted', zero_division=0),
        'overall_recall': recall_score(all_labels, all_predictions, 
                                    average='weighted', zero_division=0)
    })
    
    # Per-class metrics
    for label in range(7):  # 0 through 6
        label_mask = np.array(all_labels) == label
        if np.any(label_mask):
            metrics[f'class_{label}_f1'] = f1_score(
                np.array(all_labels) == label,
                np.array(all_predictions) == label,
                zero_division=0
            )
    
    # Overlap metrics
    if len(overlap_predictions) > 0:
        metrics.update({
            'overlap_f1': f1_score(overlap_labels, overlap_predictions, 
                                average='weighted', zero_division=0),
            'overlap_precision': precision_score(overlap_labels, overlap_predictions, 
                                            average='weighted', zero_division=0),
            'overlap_recall': recall_score(overlap_labels, overlap_predictions, 
                                        average='weighted', zero_division=0)
        })
    
    # Add confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Add per-class precision and recall
    for label in range(7):
        label_mask = np.array(all_labels) == label
        if np.any(label_mask):
            metrics[f'class_{label}_precision'] = precision_score(
                np.array(all_labels) == label,
                np.array(all_predictions) == label,
                zero_division=0
            )
            metrics[f'class_{label}_recall'] = recall_score(
                np.array(all_labels) == label,
                np.array(all_predictions) == label,
                zero_division=0
            )
    
    return metrics, conf_matrix

### Setting up datasets and models for various tasks ###

# Load and prepare dataset
def setup_data(config, train_path, test_path=None, tokenizer=None, tagging_scheme='joint'):
    """
    Set up data loaders for training, validation, and/or test data.        
    Args:
        config (dict): Configuration dictionary
        train_path (str): Path to training data
        test_path (str, optional): Path to test data
        tokenizer: The tokenizer to use
        tagging_scheme (str): The tagging scheme to use ('joint' or other schemes)        
    Returns:
        tuple: If test_path is None:
            (train_dataloader, val_dataloader, label2id, id2label)            If test_path is provided:
            (train_dataloader, val_dataloader, test_dataloader, label2id, id2label)
    """
    # Prepare training data first to establish label mappings
    train_documents, train_label_ids, label2id, id2label = prepare_data(
        train_path, 
        tagging_scheme=tagging_scheme
    )
        
    # Create dataset for training data
    train_full_dataset = MyDataset(train_documents, train_label_ids, tokenizer)        
    # Split into train and validation
    train_size = int(config['train_split'] * len(train_full_dataset))
    val_size = len(train_full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_full_dataset, 
        [train_size, val_size]
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True        )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size']        )        
    if test_path:
    # Prepare test data using the same label mappings
        test_documents, test_label_ids, _, _ = prepare_data(
            test_path, 
            tagging_scheme=tagging_scheme,
            label2id=label2id  # Use existing label mappings
            )
        test_dataset = MyDataset(test_documents, test_label_ids, tokenizer)
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=config['batch_size']
        )
        return train_dataloader, val_dataloader, test_dataloader, label2id, id2label
        
    return train_dataloader, val_dataloader, label2id, id2label

# Model set-up function
def setup_model(config):
    """
    Set up the model, tokenizer, and device based on configuration.
    
    Args:
        config (dict): Configuration dictionary containing:
            - model_name: Name/path of the pre-trained model
            - model_type: Type of model ('distilbert', 'xlmroberta', or 'longformer')
            - num_labels: Number of labels for classification
            - max_length: Maximum sequence length
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    model_mapping = {
        'distilbert': (DistilBertForTokenClassification, DistilBertTokenizerFast),
        'xlmroberta': (XLMRobertaForTokenClassification, XLMRobertaTokenizerFast),
        'longformer': (LongformerForTokenClassification, LongformerTokenizerFast)
    }
    # Get the appropriate model and tokenizer classes
    if config['model_type'] not in model_mapping:
        raise ValueError(f"Unsupported model type: {config['model_type']}")
        
    ModelClass, TokenizerClass = model_mapping[config['model_type']]
        
    # Initialize tokenizer and model
    try:
        tokenizer = TokenizerClass.from_pretrained(
            config['model_name'],
            max_length=config['max_length'],
            is_split_into_words=True,
            truncation=True
        )
            
        model = ModelClass.from_pretrained(
            config['model_name'],
            num_labels=config['num_labels']
        )
    except Exception as e:
        raise Exception(f"Error loading model or tokenizer: {str(e)}")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    return model, tokenizer, device

def main():
    # Get configuration
    num_labels = 7  # Number of labels for joint task
    model_type = 'distilbert'  # or 'xlmroberta' or 'longformer'
    config = get_model_config(model_type, num_labels)

    # Set up model and get tokenizer
    model, tokenizer, device = setup_model(config)

    # Load and prepare data
    train_dataloader, val_dataloader, label2id, id2label = setup_data(
        config, 
        CORPUS_PATHS['MAIN_PATH'], 
        tokenizer=tokenizer
    )

    # Train model
    trained_model = train_model(
        model, 
        train_dataloader, 
        val_dataloader, 
        config['num_epochs'], 
        device, 
        initial_lr=config['learning_rate'],  # Note: changed from initial_lr to learning_rate
        patience=config['early_stopping_patience']  # Note: changed from patience to early_stopping_patience
    )

    # Save model with model type in the filename
    model_filename = f"../models/{model_type}_{num_labels}labels.pt"
    model.save_pretrained(model_filename)
    # Evaluate model
    metrics, conf_matrix = evaluate_model(trained_model, val_dataloader, device)

    # Print evaluation metrics
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(conf_matrix)

if __name__ == "__main__":
    main()

import pdb
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", message="LoadLib.*failed")
from transformers import (
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizerFast,
    LongformerForTokenClassification,
    LongformerTokenizerFast,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    AutoModelForTokenClassification
)
from config import *
import json
import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'garbage_collection_threshold:0.8,max_split_size_mb:128'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch import nn
torch.cuda.empty_cache()
torch.cuda.memory.set_per_process_memory_fraction(0.95)
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import math
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from sklearn.model_selection import train_test_split
from bio_encoder import process_corpus
import argparse
import gc
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_

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

# Custom collateFN
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

### Prepare DFM and Gina models for token classification ###

class DFMForTokenClassification(nn.Module):
    def __init__(self, num_labels, model_name="KennethEnevoldsen/dfm-sentence-encoder-large-exp2-no-lang-align"): # Or dfm-sentence-encoder-large-medium
        super(DFMForTokenClassification, self).__init__()
        self.dfm = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Get hidden size from model config
        hidden_size = self.dfm.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.dfm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if 'token_type_ids' in kwargs else None,
        )

        # Get the token embeddings (first element of model output)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.classifier.out_features)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return {
            'loss': loss,            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None,
        }

    # Helper function for mean pooling if needed
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class GinaForTokenClassification(nn.Module):
    def __init__(self, num_labels, model_name="jinaai/jina-embeddings-v3"):
        super(GinaForTokenClassification, self).__init__()
        self.gina = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.gina.config.hidden_size, num_labels)
        
        # Convert all model parameters to float32
        self.gina.to(torch.float32)
        self.classifier.to(torch.float32)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Ensure inputs are float32
        input_ids = input_ids.to(torch.long)
        attention_mask = attention_mask.to(torch.float32)
        
        outputs = self.gina(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0].to(torch.float32)  # Ensure float32
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.classifier.out_features)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None,
        }

    def encode(self, texts, task="classification", max_length=None, truncate_dim=None):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        with torch.no_grad():
            outputs = self.gina(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        if truncate_dim:
            embeddings = embeddings[:, :truncate_dim]
        return embeddings.to(torch.float32)  # Ensure float32
 
### Model and training config ###

def get_model_config(model_type, tagging_scheme):
    base_config = {
        'batch_size': 12,
        'gradient_accumulation_steps': 4,
        'num_epochs': 50,
        'learning_rate': 5e-3,
        'train_split': 0.8,
        'early_stopping_patience': 5,
        'tagging_scheme': tagging_scheme,
        'num_labels': 7 if tagging_scheme == 'joint' else 3,
        'weight_decay': 0.01,  # Added weight decay for regularization
        'dropout': 0.2,  # Increased dropout for regularization
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
            'max_length': 1024,
        },
        'dfm': {
            'model_type': 'dfm',
            'model_name': 'KennethEnevoldsen/dfm-sentence-encoder-large-exp2-no-lang-align', # or 'KennethEnevoldsen/dfm-sentence-encoder-medium'
            'max_length': 512,  # Adjust this based on DFM's capabilities
        },
        'eurobert': {
            'model_type': 'eurobert',
            'model_name': 'EuroBERT/EuroBERT-210M',  # or 'EuroBERT/EuroBERT-610M'
            'max_length': 1024,
        },
        'gina': {
            'model_type': 'gina',
            'model_name': 'jinaai/jina-embeddings-v3',
            'max_length': 1024,
        },
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

    # Assert that the number of labels matches the expected number based on the tagging scheme
    expected_num_labels = 7 if tagging_scheme == 'joint' else 3
    assert len(label2id) == expected_num_labels, f"Expected {expected_num_labels} labels for {tagging_scheme} scheme, but got {len(label2id)}"

    # Convert labels to IDs
    label_ids = [[label2id[label] for label in doc_labels] for doc_labels in labels]

    # Print some basic statistics
    print(f"\nCorpus Statistics for {corpus_path}:")
    print(f"Number of documents: {len(documents)}")
    print(f"Number of unique labels in this corpus: {len(label2id)}")

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
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.processed_texts = []
        self.processed_labels = []
        self.chunk_to_doc_mapping = []
        self.encodings = []

        for doc_idx, (text, label) in enumerate(zip(texts, labels)):
            if len(text) > max_len:
                chunk_texts, chunk_labels = split_long_document(text, label, max_len)
                self.processed_texts.extend(chunk_texts)
                self.processed_labels.extend(chunk_labels)
                self.chunk_to_doc_mapping.extend([doc_idx] * len(chunk_texts))
            else:
                self.processed_texts.append(text)
                self.processed_labels.append(label)
                self.chunk_to_doc_mapping.append(doc_idx)

        self.tokenizer = tokenizer
        self.max_len = max_len

        # Pre-encode all texts
        for text in self.processed_texts:
            encoding = self.tokenizer(text, 
                                      is_split_into_words=True, 
                                      max_length=self.max_len, 
                                      truncation=True, 
                                      padding='max_length')
            self.encodings.append(encoding)

        print(f"\nDataset Statistics:")
        print(f"Total documents: {len(set(self.chunk_to_doc_mapping))}")
        print(f"Total chunks: {len(self.processed_texts)}")

    def __getitem__(self, idx):
        encoding = self.encodings[idx]
        label = self.processed_labels[idx]

        word_ids = encoding.word_ids()
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(label_ids)
        }

    def __len__(self):
        return len(self.processed_texts)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001, monitor='val_loss', mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.monitor = monitor
        self.mode = mode  # 'min' for loss, 'max' for metrics like accuracy or F1

    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
        elif self.mode == 'min':
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == 'max':
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Training loop function with early stopping and learning rate scheduler
def train_model(model, train_dataloader, val_dataloader, num_epochs, device,
                initial_lr=5e-5, patience=5, monitor='macro_f1',
                gradient_accumulation_steps=4, label2id=None):

    scaler = GradScaler()

    # Calculate class weights from all training data
    train_labels = []
    for batch in train_dataloader:
        labels = batch['labels']
        train_labels.extend(labels[labels != -100].cpu().numpy())

    # Compute class weights
    unique_classes, class_counts = np.unique(train_labels, return_counts=True)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)

    # Initialize Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    focal_loss = FocalLoss(gamma=2.0, alpha=class_weights)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001, monitor='macro_f1', mode='max')

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)

    # Initialize learning rate scheduler
    scheduler = OneCycleLR(optimizer, max_lr=initial_lr, epochs=num_epochs, steps_per_epoch=len(train_dataloader))

    best_metric = 0
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0

        for i, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                ce_loss = criterion(logits.view(-1, len(label2id)), labels.view(-1))
                fl_loss = focal_loss(logits.view(-1, len(label2id)), labels.view(-1))
                loss = (ce_loss + fl_loss) / 2  # Combine CrossEntropyLoss and FocalLoss
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_train_loss += loss.item() * gradient_accumulation_steps

            # Explicit tensor cleanup
            del input_ids, attention_mask, labels, outputs, logits
            torch.cuda.empty_cache()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                loss = criterion(logits.view(-1, len(label2id)), labels.view(-1))
                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds[labels != -100].cpu().numpy())
                all_labels.extend(labels[labels != -100].cpu().numpy())

                # Explicit tensor cleanup
                del input_ids, attention_mask, labels, outputs, logits
                torch.cuda.empty_cache()

        avg_val_loss = total_val_loss / len(val_dataloader)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        # Print progress and memory usage
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average validation loss: {avg_val_loss:.4f}")
        print(f"Macro F1 Score: {macro_f1:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached GPU memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        # Early stopping check
        if early_stopping(macro_f1):
            print("Early stopping triggered!")
            break

        # Save best model
        if macro_f1 > best_metric:
            best_metric = macro_f1
            torch.save(model.state_dict(), f"best_model.pt")
            print("New best model saved!")

        gc.collect()
        torch.cuda.empty_cache()

    # Load best model state
    if os.path.exists("best_model.pt"):
        model.load_state_dict(torch.load("best_model.pt"))

    return model

# evaluation metrics for split-aware learning
def evaluate_model(model, dataloader, device, num_labels, id2label, max_len):
    model.eval()
    all_predictions = []
    all_labels = []
    overlap_predictions = []
    overlap_labels = []

    overlap_size = max_len // 2

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
            predictions = torch.argmax(logits, dim=-1)

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
    for label in range(num_labels): # Per class
        label_name = id2label[label]
        label_mask = np.array(all_labels) == label
        if np.any(label_mask):
            metrics[f'class_{label_name}_f1'] = f1_score(
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
    for label in range(num_labels):
        label_name = id2label[label]
        label_mask = np.array(all_labels) == label
        if np.any(label_mask):
            metrics[f'class_{label_name}_precision'] = precision_score(
                np.array(all_labels) == label,
                np.array(all_predictions) == label,
                zero_division=0
            )
            metrics[f'class_{label_name}_recall'] = recall_score(
                np.array(all_labels) == label,
                np.array(all_predictions) == label,
                zero_division=0
            )

    return metrics, conf_matrix

### Setting up datasets and models for various tasks ###

# Load and prepare dataset
def setup_data(config, train_path, test_paths=None, tokenizer=None):
    """
    Set up data loaders for training, validation, and test data.

    Args:
        config (dict): Configuration dictionary
        train_path (str): Path to training data
        test_paths (str or list, optional): Single test path or list of test paths
        tokenizer: The tokenizer to use
        tagging_scheme (str): The tagging scheme to use ('joint' or other schemes)

    Returns:
        tuple: If test_paths is None:
            (train_dataloader, val_dataloader, label2id, id2label, num_labels)
        If single test_path is provided:
            (train_dataloader, val_dataloader, test_dataloader, label2id, id2label, num_labels)
        If multiple test_paths are provided:
            (train_dataloader, val_dataloader, list_of_test_dataloaders, label2id, id2label)
    """

    # Ensure tagging_scheme is in the config
    tagging_scheme = config.get('tagging_scheme', 'joint') # Get tagging scheme from config

    # Prepare training data first to establish label mappings
    train_documents, train_label_ids, label2id, id2label = prepare_data(
        train_path, 
        tagging_scheme=config['tagging_scheme']
    )

    # Create dataset for training data
    max_len = config['max_length']
    full_dataset = MyDataset(train_documents, train_label_ids, tokenizer, max_len=max_len)

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(config['train_split'] * total_size)
    val_size = total_size - train_size

    # Perform random split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=8
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=8
    )

    if test_paths is None:
        return train_dataloader, val_dataloader, label2id, id2label

    # Handle test data
    if isinstance(test_paths, str):
        # Single test path
        test_documents, test_label_ids, _, _ = prepare_data(
            test_paths, 
            tagging_scheme=tagging_scheme,
            label2id=label2id
        )
        test_dataset = MyDataset(test_documents, test_label_ids, tokenizer, max_len=max_len)
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=8
        )
        return train_dataloader, val_dataloader, test_dataloader, label2id, id2label

    else:
        # Multiple test paths
        test_dataloaders = []
        for test_path in test_paths:
            test_documents, test_label_ids, _, _ = prepare_data(
                test_path, 
                tagging_scheme=tagging_scheme,
                label2id=label2id
            )
            test_dataset = MyDataset(test_documents, test_label_ids, tokenizer, max_len=max_len)
            test_dataloader = DataLoader(
                test_dataset, 
                batch_size=config['batch_size'],
                collate_fn=collate_fn,
                pin_memory=True,
                num_workers=8
            )
            test_dataloaders.append(test_dataloader)

        return train_dataloader, val_dataloader, test_dataloaders, label2id, id2label

# Model set-up function
def setup_model(config):
    model_mapping = {
        'distilbert': (DistilBertForTokenClassification, DistilBertTokenizerFast),
        'xlmroberta': (XLMRobertaForTokenClassification, XLMRobertaTokenizerFast),
        'longformer': (LongformerForTokenClassification, LongformerTokenizerFast),
        'dfm': (DFMForTokenClassification, AutoTokenizer),
        'eurobert': (AutoModelForTokenClassification, AutoTokenizer),
        'gina': (GinaForTokenClassification, AutoTokenizer)
    }
    # Get the appropriate model and tokenizer classes
    if config['model_type'].lower() not in model_mapping:
        raise ValueError(f"Unsupported model type: {config['model_type']}")

    ModelClass, TokenizerClass = model_mapping[config['model_type'].lower()]

    # Initialize tokenizer
    try:
        # Tokenizer initialization
        tokenizer_kwargs = {
            'max_length': config['max_length'],
            'truncation': True,
            'is_split_into_words': True
        }
        if config['model_type'] in ['dfm', 'eurobert', 'gina']:
            tokenizer_kwargs = {'trust_remote_code': True}
        elif config['model_type'] == 'longformer':
            tokenizer_kwargs['add_prefix_space'] = True

        tokenizer = TokenizerClass.from_pretrained(config['model_name'], **tokenizer_kwargs)

        # Initialize model
        model_kwargs = {
            'num_labels': config['num_labels'],
        }

        if config['model_type'] == 'gina':
            model = ModelClass(num_labels=config['num_labels'], model_name=config['model_name'])
        elif config['model_type'] == 'dfm':
            model = ModelClass(model_name=config['model_name'], **model_kwargs)
        else:
            if config['model_type'] == 'eurobert':
                model_kwargs['trust_remote_code'] = True
            model = ModelClass.from_pretrained(config['model_name'], **model_kwargs)

        # Adjust model config for max_length if applicable
        if hasattr(model, 'config') and hasattr(model.config, 'max_position_embeddings'):
            model.config.max_position_embeddings = config['max_length']

    except Exception as e:
        raise Exception(f"Error loading model or tokenizer: {str(e)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return model, tokenizer, device

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate models with different configurations')

    # Model configuration
    parser.add_argument('--model_type', type=str, default='dfm', 
                        choices=['distilbert', 'xlmroberta', 'longformer', 'dfm', 'eurobert', 'gina'],
                        help='Type of model to use)')
    parser.add_argument('--tagging_scheme', type=str, default='joint',
                        choices=['joint', 'separate', 'metaphor', 'hyperbole'],
                        help='Tagging scheme for task (default: joint)')

    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (default: use config default)')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of training epochs (default: use config default)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (default: use config default)')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Maximum sequence length (default: use config default)')
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping patience (default: use config default)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of steps for gradient accumulation')

    # Data paths
    parser.add_argument('--train_path', type=str, default=None,
                        help='Path to main corpus (default: use CORPUS_PATHS["MAIN_PATH"])')
    parser.add_argument('--test_paths', nargs='+', default=None,
                        help='Paths to test annotation files (default: use ann_paths values)')

    # Output options
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='Directory to save the model (default: ../models)')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the model after training')
    parser.add_argument('--run_number', type=int, default=None,
                        help='Run number for multiple iterations')

    return parser.parse_args()

def generate_report(config, val_metrics, test_metrics, label2id):
    report = {
        "model_type": config['model_type'],
        "tagging_scheme": config['tagging_scheme'],
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "label_mapping": label2id
    }
    return report

def main(run_number=None):
    # Parse command line arguments
    args = parse_args()

    if run_number is not None:
        print(f"Starting run number {run_number}")
    
    # Create config with default values including num_labels
    config = get_model_config(args.model_type, args.tagging_scheme)
    
    # Override config with command line arguments if provided
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.max_length is not None:
        config['max_length'] = args.max_length
    if args.patience is not None:
        config['early_stopping_patience'] = args.patience
    if args.gradient_accumulation_steps is not None:
        config['gradient_accumulation_steps'] = args.gradient_accumulation_steps

    # Determine paths
    train_path = args.train_path if args.train_path else CORPUS_PATHS['MAIN_PATH']
    test_paths = args.test_paths if args.test_paths else [CORPUS_PATHS['REANNO_PATH'], ann_paths['CONSENSUS_ANN1_PATH'], ann_paths['CONSENSUS_ANN2_PATH']]

    # Set up model and get tokenizer
    model, tokenizer, device = setup_model(config)

    # Get data
    train_dataloader, val_dataloader, test_dataloaders, label2id, id2label = setup_data(
        config, 
        train_path,
        test_paths=test_paths,
        tokenizer=tokenizer
    )
    
    # Print final configuration
    print(f"Configuration: {config}")

    # Train model
    trained_model = train_model(
        model, 
        train_dataloader, 
        val_dataloader, 
        config['num_epochs'], 
        device, 
        initial_lr=config['learning_rate'],
        patience=config['early_stopping_patience'],
        label2id=label2id,
        gradient_accumulation_steps=config['gradient_accumulation_steps']
    )

    # Save model with model type in the filename
    if args.save_model:
        os.makedirs(args.output_dir, exist_ok=True)
        model_filename = f"{args.output_dir}/{args.model_type}_{args.num_labels}labels.pt"
        model_filename = f"{args.output_dir}/{args.model_type}_{args.num_labels}labels"
        print(f"Saving model to {model_filename}")
        model.save_checkpoint(model_filename)

    # Evaluate model on validation data
    val_metrics, val_conf_matrix = evaluate_model(
        trained_model, 
        val_dataloader, 
        device,
        config['num_labels'], 
        id2label, 
        config['max_length']
    )

    # Print validation metrics
    print("Validation Metrics:")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nValidation Confusion Matrix:")
    print(val_conf_matrix)

    # Evaluate on all test sets and calculate average metrics
    all_test_metrics = []
    print("\nTest Set Evaluations:")

    for i, test_dataloader in enumerate(test_dataloaders, 1):
        test_metrics, test_conf_matrix = evaluate_model(
            trained_model, 
            test_dataloader, 
            device, 
            config['num_labels'], 
            id2label, 
            config['max_length']
        )
        all_test_metrics.append(test_metrics)

        print(f"\nTest Set {i} Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")

        print(f"\nTest Set {i} Confusion Matrix:")
        print(test_conf_matrix)

    # Calculate and print average metrics across test sets
    avg_metrics = {}
    for metric in all_test_metrics[0].keys():
        avg_value = sum(tm[metric] for tm in all_test_metrics) / len(all_test_metrics)
        avg_metrics[metric] = avg_value
        print(f"Average {metric}: {avg_value:.4f}")

    # Print label mapping
    print("\nLabel Mapping:")
    for label, id in label2id.items():
        print(f"{label}: {id}") 

    # Generate and save report
    if run_number is not None:
        report_filename = f"{args.output_dir}/{args.model_type}_{args.tagging_scheme}_run{run_number}_report.json"
    else:
        report_filename = f"{args.output_dir}/{args.model_type}_{args.tagging_scheme}_report.json"
    
    report = generate_report(config, val_metrics, all_test_metrics[0], label2id)
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_filename}")

if __name__ == "__main__":
    args = parse_args()
    run_number = args.run_number if hasattr(args, 'run_number') else None
    main(run_number)

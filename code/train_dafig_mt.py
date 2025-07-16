import pdb
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from train_dafig import collate_fn
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from config import *
import argparse
from datetime import datetime
import json
import os
from typing import Dict, List, Tuple
from bio_encoder import process_corpus

# Variables
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

### Data pre-processing ###

def prepare_data(corpus_path: str, label2id: Dict[str, int] = None) -> Tuple[List[List[str]], List[List[int]], List[List[int]], Dict[str, int], Dict[int, str]]:
    """
    Prepare data for multi-task learning with metaphor and hyperbole tasks.

    Args:
        corpus_path (str): Path to the corpus
        label2id (dict, optional): Existing label mapping. If None, a new mapping will be created

    Returns:
        documents (list): List of tokenized documents
        met_label_ids (list): List of label IDs for metaphor task
        hyp_label_ids (list): List of label IDs for hyperbole task
        label2id (dict): Mapping of labels to IDs
        id2label (dict): Mapping of IDs to labels
    """
    # Get tagged documents from single corpus
    data = process_corpus(corpus_path, tagging_scheme='separate')

    # Process the data
    documents = []
    met_labels = []
    hyp_labels = []

    for doc_id, doc_data in data.items():
        doc_tokens = []
        doc_met_labels = []
        doc_hyp_labels = []
        
        # Assuming 'metaphor' and 'hyperbole' have the same tokens in the same order
        for (token, met_label), (_, hyp_label) in zip(doc_data['metaphor'], doc_data['hyperbole']):
            doc_tokens.append(token)
            doc_met_labels.append(met_label)
            doc_hyp_labels.append(hyp_label)
        
        documents.append(doc_tokens)
        met_labels.append(doc_met_labels)
        hyp_labels.append(doc_hyp_labels)

    # Create or update label mappings
    if label2id is None:
        unique_labels = set([label for doc_labels in met_labels + hyp_labels for label in doc_labels])
        label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Convert labels to IDs
    met_label_ids = [[label2id[label] for label in doc_labels] for doc_labels in met_labels]
    hyp_label_ids = [[label2id[label] for label in doc_labels] for doc_labels in hyp_labels]

    # Print some basic statistics
    print(f"\nCorpus Statistics for {corpus_path}:")
    print(f"Number of documents: {len(documents)}")
    print(f"Number of unique labels: {len(label2id)}")

    return documents, met_label_ids, hyp_label_ids, label2id, id2label

def split_long_document(text: List[str], met_labels: List[int], hyp_labels: List[int], max_len: int) -> Tuple[List[List[str]], List[List[int]], List[List[int]]]:
    """
    Create overlapping chunks for long documents.

    Args:
        text (List[str]): List of tokens in the document
        met_labels (List[int]): List of metaphor label IDs
        hyp_labels (List[int]): List of hyperbole label IDs
        max_len (int): Maximum length of each chunk

    Returns:
        chunks_text (List[List[str]]): List of chunked texts
        chunks_met_labels (List[List[int]]): List of chunked metaphor labels
        chunks_hyp_labels (List[List[int]]): List of chunked hyperbole labels
    """
    chunks_text = []
    chunks_met_labels = []
    chunks_hyp_labels = []
    stride = max_len // 2  # 50% overlap

    for i in range(0, len(text), stride):
        chunk_text = text[i:i + max_len]
        chunk_met_labels = met_labels[i:i + max_len]
        chunk_hyp_labels = hyp_labels[i:i + max_len]
        if len(chunk_text) > max_len // 4:  # Only keep chunks with substantial content
            chunks_text.append(chunk_text)
            chunks_met_labels.append(chunk_met_labels)
            chunks_hyp_labels.append(chunk_hyp_labels)

    return chunks_text, chunks_met_labels, chunks_hyp_labels

# Dataset creation
class MultiTaskDataset(Dataset):
    def __init__(self, texts, met_labels, hyp_labels, tokenizer, max_len=512):
        self.processed_texts = []
        self.processed_met_labels = []
        self.processed_hyp_labels = []
        self.chunk_to_doc_mapping = []
        self.encodings = []

        for doc_idx, (text, met_label, hyp_label) in enumerate(zip(texts, met_labels, hyp_labels)):
            if len(text) > max_len:
                chunk_texts, chunk_met_labels, chunk_hyp_labels = split_long_document(text, met_label, hyp_label, max_len)
                self.processed_texts.extend(chunk_texts)
                self.processed_met_labels.extend(chunk_met_labels)
                self.processed_hyp_labels.extend(chunk_hyp_labels)
                self.chunk_to_doc_mapping.extend([doc_idx] * len(chunk_texts))
            else:
                self.processed_texts.append(text)
                self.processed_met_labels.append(met_label)
                self.processed_hyp_labels.append(hyp_label)
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
        met_label = self.processed_met_labels[idx]
        hyp_label = self.processed_hyp_labels[idx]

        word_ids = encoding.word_ids()
        met_label_ids = [-100 if word_id is None else met_label[word_id] for word_id in word_ids]
        hyp_label_ids = [-100 if word_id is None else hyp_label[word_id] for word_id in word_ids]

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'met_labels': torch.tensor(met_label_ids),
            'hyp_labels': torch.tensor(hyp_label_ids)
        }

    def __len__(self):
        return len(self.processed_texts)

### Model setup ###

# Multi-task learning model using DFM-Sentence-Encoder-Large-Exp2-no-lang-align model and cross-entropy loss for both tasks
class DFMForMultiTaskLearning(nn.Module):
    def __init__(self, num_labels_task1, num_labels_task2, model_name="KennethEnevoldsen/dfm-sentence-encoder-large-exp2-no-lang-align"):
        super(DFMForMultiTaskLearning, self).__init__()
        self.dfm = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden_size = self.dfm.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier_task1 = nn.Linear(hidden_size, num_labels_task1)
        self.classifier_task2 = nn.Linear(hidden_size, num_labels_task2)

    def forward(self, input_ids, attention_mask, task_id, labels=None):
        outputs = self.dfm(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        if task_id == 1:
            logits = self.classifier_task1(sequence_output)
        elif task_id == 2:
            logits = self.classifier_task2(sequence_output)
        else:
            raise ValueError("Invalid task_id")

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

        return {'loss': loss, 'logits': logits}

def train_model(model, train_dataloader, val_dataloader, num_epochs, device, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    best_model = None
    best_f1 = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            met_labels = batch['met_labels'].to(device)
            hyp_labels = batch['hyp_labels'].to(device)

            optimizer.zero_grad()

            # Forward pass for task 1
            outputs_task1 = model(input_ids, attention_mask, task_id=1, labels=met_labels)
            loss_task1 = outputs_task1['loss']

            # Forward pass for task 2
            outputs_task2 = model(input_ids, attention_mask, task_id=2, labels=hyp_labels)
            loss_task2 = outputs_task2['loss']

            # Combine losses
            loss = loss_task1 + loss_task2
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average training loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_metrics = evaluate_model(model, val_dataloader, device)
        val_f1_macro = (val_metrics['metaphor']['f1_macro'] + val_metrics['hyperbole']['f1_macro']) / 2
        print(f"Validation Macro F1: {val_f1_macro:.4f}")

        # Check if this is the best model so far
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            best_model = model.state_dict().copy()

        # Early stopping
        early_stopping(val_f1_macro)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def evaluate_model(model, dataloader, device):
    model.eval()
    all_met_preds = []
    all_met_labels = []
    all_hyp_preds = []
    all_hyp_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            met_labels = batch['met_labels'].to(device)
            hyp_labels = batch['hyp_labels'].to(device)

            # Metaphor task
            outputs_met = model(input_ids, attention_mask, task_id=1)
            logits_met = outputs_met['logits']
            preds_met = torch.argmax(logits_met, dim=-1)

            # Hyperbole task
            outputs_hyp = model(input_ids, attention_mask, task_id=2)
            logits_hyp = outputs_hyp['logits']
            preds_hyp = torch.argmax(logits_hyp, dim=-1)

            # Collect predictions and labels
            all_met_preds.extend(preds_met.cpu().numpy().flatten())
            all_met_labels.extend(met_labels.cpu().numpy().flatten())
            all_hyp_preds.extend(preds_hyp.cpu().numpy().flatten())
            all_hyp_labels.extend(hyp_labels.cpu().numpy().flatten())

    # Remove padding (-100) from labels and predictions
    all_met_preds = [pred for pred, label in zip(all_met_preds, all_met_labels) if label != -100]
    all_met_labels = [label for label in all_met_labels if label != -100]
    all_hyp_preds = [pred for pred, label in zip(all_hyp_preds, all_hyp_labels) if label != -100]
    all_hyp_labels = [label for label in all_hyp_labels if label != -100]

    # Calculate metrics for metaphor task
    met_precision = precision_score(all_met_labels, all_met_preds, average='weighted', zero_division=0)
    met_recall = recall_score(all_met_labels, all_met_preds, average='weighted', zero_division=0)
    met_f1_weighted = f1_score(all_met_labels, all_met_preds, average='weighted', zero_division=0)
    met_f1_macro = f1_score(all_met_labels, all_met_preds, average='macro', zero_division=0)
    met_report = classification_report(all_met_labels, all_met_preds, zero_division=0)

    # Calculate metrics for hyperbole task
    hyp_precision = precision_score(all_hyp_labels, all_hyp_preds, average='weighted', zero_division=0)
    hyp_recall = recall_score(all_hyp_labels, all_hyp_preds, average='weighted', zero_division=0)
    hyp_f1_weighted = f1_score(all_hyp_labels, all_hyp_preds, average='weighted', zero_division=0)
    hyp_f1_macro = f1_score(all_hyp_labels, all_hyp_preds, average='macro', zero_division=0)
    hyp_report = classification_report(all_hyp_labels, all_hyp_preds, zero_division=0)

    return {
        'metaphor': {
            'precision': met_precision, 
            'recall': met_recall, 
            'f1_weighted': met_f1_weighted,
            'f1_macro': met_f1_macro,
            'report': met_report
        },
        'hyperbole': {
            'precision': hyp_precision, 
            'recall': hyp_recall, 
            'f1_weighted': hyp_f1_weighted,
            'f1_macro': hyp_f1_macro,
            'report': hyp_report
        }
    }

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_size', type=float, default=0.8)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_len', type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    train_path = CORPUS_PATHS['MAIN_PATH']
    test_path = CORPUS_PATHS['REANNO_PATH']
    
    # Use autotokenizer for tokenization of pre-tokenized outputs
    tokenizer = AutoTokenizer.from_pretrained('KennethEnevoldsen/dfm-sentence-encoder-large-exp2-no-lang-align', truncation=True, is_split=True)
    
    # Set up datasets
    train_docs, train_met_label_ids, train_hyp_label_ids, label2id, id2label = prepare_data(train_path)
    test_docs, test_met_label_ids, test_hyp_label_ids, _, _ = prepare_data(test_path, label2id=label2id)
    
    # Create datasets
    train_dataset = MultiTaskDataset(train_docs, train_met_label_ids, train_hyp_label_ids, tokenizer, args.max_len)
    test_dataset = MultiTaskDataset(test_docs, test_met_label_ids, test_hyp_label_ids, tokenizer, args.max_len)
    
    # Calculate split sizes
    total_size = len(train_dataset)
    train_size = int(args.split_size * total_size)
    val_size = total_size - train_size

    # Perform random split
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=8
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=8
    )

    # Define model
    model = DFMForMultiTaskLearning(len(label2id), len(label2id), model_name='KennethEnevoldsen/dfm-sentence-encoder-large-exp2-no-lang-align')
    model = model.to(device)
    
    # Train the model
    model = train_model(model, train_dataloader, val_dataloader, args.num_epochs, device, args.learning_rate)

    # Evaluate the model
    # Evaluate the model
    print("Evaluating model on Metaphor Task...")
    task1_metrics = evaluate_model(model, test_dataloader, device)['metaphor']
    print(f"Precision: {task1_metrics['precision']:.4f}, Recall: {task1_metrics['recall']:.4f}")
    print(f"F1 (Weighted): {task1_metrics['f1_weighted']:.4f}, F1 (Macro): {task1_metrics['f1_macro']:.4f}")
    print("Detailed Metaphor Classification Report:")
    print(task1_metrics['report'])
    
    print("\nEvaluating model on Hyperbole Task...")
    task2_metrics = evaluate_model(model, test_dataloader, device)['hyperbole']
    print(f"Precision: {task2_metrics['precision']:.4f}, Recall: {task2_metrics['recall']:.4f}")
    print(f"F1 (Weighted): {task2_metrics['f1_weighted']:.4f}, F1 (Macro): {task2_metrics['f1_macro']:.4f}")
    print("Detailed Hyperbole Classification Report:")
    print(task2_metrics['report'])

    # Write reports to json file
    with open('../results/mt_report.json', 'w') as f:
        json.dump({
            'task1_metrics': task1_metrics,
            'task2_metrics': task2_metrics
        }, f, indent=4)
    
    # Save the trained model
    #torch.save(model.state_dict(), './models/trained_model.pth')

if __name__ == "__main__":
    main()
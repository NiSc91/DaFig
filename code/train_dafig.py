from config import *
import json
import torch
import numpy as np
#import torch_directml
from transformers import AutoTokenizer, DistilBertForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split
import math
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn

# Load the JSON file
with open(os.path.join(TEMP_DIR, 'tagged_documents.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

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
    for label in range(9):  # 0 through 8
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
    for label in range(9):
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

### Preprocess the data ###
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

# Create label to ID mapping
unique_labels = set([label for doc_labels in labels for label in doc_labels])
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# Convert labels to IDs
label_ids = [[label2id[label] for label in doc_labels] for doc_labels in labels]

# After creating labels
print("\nLabel Statistics before dataset creation:")
print(f"Number of documents: {len(labels)}")
print("Sample of labels from first document:")
print(f"Length: {len(labels[0])}")
print(f"Unique values: {set(labels[0])}")
print(f"Number of -100 labels: {sum(1 for l in labels[0] if l == -100)}")

### Dataset and data loader ###
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = DistilBertForTokenClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=len(label2id))

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
        
        # Debug print for first item
        if idx == 0:
            print("\nSample label processing:")
            print(f"Original label length: {len(label)}")
            print(f"Word IDs length: {len(word_ids)}")
            print(f"Number of -100 labels: {sum(1 for l in label if l == -100)}")
            
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(label_ids)
        }

    def __len__(self):  # Added __len__ method
        return len(self.processed_texts)

# Create dataset
dataset = MyDataset(documents, label_ids, tokenizer)

# analyze sequence lengths
print(analyze_sequence_lengths(dataset))

# Hyperparameters
num_epochs = 10
batch_size = 16
learning_rate = 5e-5
weight_decay = 0.01

# Split dataset into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Get a random sample from the dataset
sample_idx = torch.randint(len(dataset), size=(1,)).item()
sample = dataset[sample_idx]

# Print the sample
print("Sample from the dataset:")
print(f"Input IDs shape: {sample['input_ids'].shape}")
print(f"Attention Mask shape: {sample['attention_mask'].shape}")
print(f"Labels shape: {sample['labels'].shape}")

# Decode the input IDs back to text
decoded_text = tokenizer.decode(sample['input_ids'])
print(f"\nDecoded text:\n{decoded_text}")

# Print the labels
print(f"\nLabels:\n{sample['labels'].tolist()}")

# Check if shapes match
assert sample['input_ids'].shape == sample['attention_mask'].shape == sample['labels'].shape, "Shapes don't match!"

# Check if the number of labels is correct
unique_labels = set(sample['labels'].tolist()) - {-100}  # Exclude padding label
print(f"\nUnique labels in this sample: {unique_labels}")
print(f"Total number of unique labels in the dataset: {len(label2id)}")

# Check dataloader
batch = next(iter(train_dataloader))
print(f"\nBatch shapes:")
print(f"Input IDs: {batch['input_ids'].shape}")
print(f"Attention Mask: {batch['attention_mask'].shape}")
print(f"Labels: {batch['labels'].shape}")

# Verify batch size
assert batch['input_ids'].shape[0] == batch_size, f"Expected batch size {batch_size}, got {batch['input_ids'].shape[0]}"

# Set up DirectML device
#dml = torch_directml.device()
#model.to(dml)

# Set up CUDA device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

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

# Train the model
model = train_model(model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    num_epochs=10,
    device=device,
    initial_lr=5e-5,  # Reduced learning rate
    patience=3        # Early stopping patience
)

# Save the best model
#model.save_pretrained("fine_tuned_distilbert_dafig")

# Evaluate the model
metrics, conf_matrix = evaluate_model(model, val_dataloader, device)

# Print evaluation results
print("\nFinal Evaluation Results:")
print(f"Overall F1 (weighted): {metrics['overall_f1_weighted']:.4f}")
print(f"Overall F1 (macro): {metrics['overall_f1_macro']:.4f}")
print(f"Overall Precision: {metrics['overall_precision']:.4f}")
print(f"Overall Recall: {metrics['overall_recall']:.4f}")

print("\nPer-class F1 scores:")
for label in range(9):
    if f'class_{label}_f1' in metrics:
        print(f"Class {label}: {metrics[f'class_{label}_f1']:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)
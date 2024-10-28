import pdb
from config import *
import json
import torch
#import torch_directml
from transformers import AutoTokenizer, DistilBertForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split

# Load the JSON file
with open(os.path.join(TEMP_DIR, 'tagged_documents.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

# Preprocess the data
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

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = DistilBertForTokenClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=len(label2id))

class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

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

# Create dataset
dataset = MyDataset(documents, label_ids, tokenizer)

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

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
num_training_steps = num_epochs * len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    
    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_distilbert_dafig")
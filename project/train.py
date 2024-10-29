import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertModel

# Load the appropriateness dataset
dataset = load_dataset('timonziegenbein/appropriateness-corpus')
print(dataset)
# Load a pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define all the labels (14 labels in total, including higher categories and subcategories)
label_columns = [
    'Inappropriateness', 'Toxic Emotions', 'Excessive Intensity', 'Emotional Deception',
    'Missing Commitment', 'Missing Seriousness', 'Missing Openness',
    'Missing Intelligibility', 'Unclear Meaning', 'Missing Relevance', 'Confusing Reasoning',
    'Detrimental Orthography', 'Reason Unclassified', 'Other Reasons'
]


# Tokenization function to prepare the data for the BERT model
def tokenize_function(examples):
    return tokenizer(examples['post_text'], padding="max_length", truncation=True, max_length=128)


# Tokenization function to prepare the data for the BERT model
def tokenize_and_encode_labels(examples):
    # Tokenize the post text
    encoding = tokenizer(examples['post_text'], padding="max_length", truncation=True)

    # Create a multi-label tensor for each example
    labels = [examples[label] for label in label_columns]
    encoding['labels'] = torch.tensor(labels, dtype=torch.float)  # Use float for multi-label loss
    return encoding


# Apply the tokenization function to the dataset
tokenized_datasets = dataset.map(tokenize_and_encode_labels, batched=False)

# Set the columns for PyTorch compatibility
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', "labels"])

# Load a pre-trained BERT model for multi-label classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_columns))

print(model)


# Define the custom compute_metrics function for multi-label classification
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits > 0).int()  # Thresholding logits to get binary predictions
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch"
)

# Trainer for fine-tuning BERT
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./appropriateness-classifier')
tokenizer.save_pretrained('./appropriateness-classifier')

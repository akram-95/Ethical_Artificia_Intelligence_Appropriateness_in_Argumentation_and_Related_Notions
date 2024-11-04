import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertModel

# Load the appropriateness dataset
dataset = load_dataset('timonziegenbein/appropriateness-corpus')
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
    # Tokenize with padding and truncation enabled
    encoding = tokenizer(
        examples['post_text'],
        padding='max_length',  # Automatically pads the sequences to the same length
        truncation=True,  # Automatically truncates longer sequences
        return_tensors='pt'  # Return PyTorch tensors
    )

    # Proceed to handle labels
    # Prepare the labels
    labels = []
    for i in range(len(examples[label_columns[0]])):  # Assuming all labels have the same length
        example_labels = [examples[label][i] for label in label_columns]
        labels.append(torch.tensor(example_labels, dtype=torch.float))

    # Pad the labels if they are of variable length
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0.0)

    # Add labels tensor to the encoding
    encoding['labels'] = labels_padded

    return encoding


# Apply the tokenization function to the dataset
tokenized_datasets = dataset.map(tokenize_and_encode_labels, batched=True)

# Set the columns for PyTorch compatibility
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', "labels"])

# Load a pre-trained BERT model for multi-label classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_columns))


# Define the custom compute_metrics function for multi-label classification
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits > 0).astype(int)  # Thresholding logits to get binary predictions
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

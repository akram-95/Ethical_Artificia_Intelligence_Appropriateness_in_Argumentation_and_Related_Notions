import torch
import logging
import warnings
import numpy as np
from torch.optim import AdamW
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support
from transformers import AutoTokenizer, DebertaForSequenceClassification, Trainer, TrainingArguments

# Defining all the labels
label_columns = [
    'Inappropriateness', 'Toxic Emotions', 'Excessive Intensity', 'Emotional Deception',
    'Missing Commitment', 'Missing Seriousness', 'Missing Openness',
    'Missing Intelligibility', 'Unclear Meaning', 'Missing Relevance', 'Confusing Reasoning',
    'Detrimental Orthography', 'Reason Unclassified', 'Other Reasons'
]

# Disabling warnings
warnings.filterwarnings("ignore")
logging.getLogger("wandb").setLevel(logging.ERROR)

# Loading dataset
dataset = load_dataset('timonziegenbein/appropriateness-corpus')

# Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')

# Tokenization function
def tokenize_and_encode_labels(examples):
    encoding = tokenizer(
        examples['post_text'],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    labels = [
        torch.tensor([examples[label][i] for label in label_columns], dtype=torch.float)
        for i in range(len(examples[label_columns[0]]))
    ]
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0.0)
    encoding['labels'] = labels_padded
    return encoding

# Applying tokenization
tokenized_datasets = dataset.map(tokenize_and_encode_labels, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Loading model
model = DebertaForSequenceClassification.from_pretrained(
    'microsoft/deberta-v3-small',
    num_labels=len(label_columns),
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True
)

# Calculating class weights dynamically
def calculate_class_weights(dataset, labels):
    label_counts = torch.zeros(len(labels))
    for example in dataset:
        label_counts += torch.tensor(example['labels'], dtype=torch.float)
    class_weights = 1.0 / (label_counts + 3e-6)
    return class_weights / class_weights.sum()

class_weights = calculate_class_weights(tokenized_datasets['train'], label_columns)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)

# Loss function with class weights
def compute_loss(logits, labels):
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    return loss_fn(logits, labels)

# Computing metrics with threshold tuning
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    thresholds = [0.4] * probs.shape[1]
    predictions = (probs > torch.tensor(thresholds)).numpy()
    #predictions = torch.round(torch.sigmoid(torch.from_numpy(logits))).numpy()
    true_labels = labels.tolist()
    pred_labels = predictions.tolist()

    precision_per_label, recall_per_label, f1_per_label = [], [], []
    for i in range(len(true_labels[0])):
        label_true = [true_labels[j][i] for j in range(len(true_labels))]
        label_pred = [pred_labels[j][i] for j in range(len(pred_labels))]
        precision, recall, f1, _ = precision_recall_fscore_support(label_true, label_pred, average='macro')
        precision_per_label.append(precision)
        recall_per_label.append(recall)
        f1_per_label.append(f1)

    avg_precision = sum(precision_per_label) / len(precision_per_label)
    avg_recall = sum(recall_per_label) / len(recall_per_label)
    avg_f1 = sum(f1_per_label) / len(f1_per_label)
    return {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1}

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    metric_for_best_model="f1",
    greater_is_better=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)

# Training the model
trainer.train()

# Saving the fine-tuned model
model.save_pretrained('./appropriateness-classifier1')
tokenizer.save_pretrained('./appropriateness-classifier1')
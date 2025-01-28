# Evaluation script
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, DebertaForSequenceClassification
print("Loading dataset...")
dataset = load_dataset('timonziegenbein/appropriateness-corpus')
## How to run this file: python3 evaluation_inappropriateness_dataset_f1_deberta.py
## this file will evaluate all labels under test split in appropriateness-corpus dataset based on the trained models which saved under: ./appropriateness-classifier
# Defining all the labels
label_columns = [
    'Inappropriateness', 'Toxic Emotions', 'Excessive Intensity', 'Emotional Deception',
    'Missing Commitment', 'Missing Seriousness', 'Missing Openness',
    'Missing Intelligibility', 'Unclear Meaning', 'Missing Relevance', 'Confusing Reasoning',
    'Detrimental Orthography', 'Reason Unclassified', 'Other Reasons'
]

print("Loading tokenizer and model...")
eval_tokenizer = AutoTokenizer.from_pretrained('./appropriateness-classifier')
eval_model = DebertaForSequenceClassification.from_pretrained('./appropriateness-classifier', ignore_mismatched_sizes=True)

def classify_texts(texts, threshold=0.3):
    all_predictions = []
    print("Classifying texts...")
    for text in texts:
        inputs = eval_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = eval_model(**inputs)
            logits = outputs.logits
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
        predictions = (probabilities >= threshold).astype(int)
        all_predictions.append(predictions)
    return all_predictions

print("Extracting test data...")
texts = [example['post_text'] for example in dataset['test']]
ground_truth = [[example[label] for label in label_columns] for example in dataset['test']]
predictions = classify_texts(texts, threshold=0.4)

print("\nCalculating Macro-F1 scores per dimension...")
f1_scores_per_dim = []
for j, dim in enumerate(label_columns):
    true_labels = [x[j] for x in ground_truth]
    pred_labels = [x[j] for x in predictions]
    scores = precision_recall_fscore_support(true_labels, pred_labels, average='macro')
    f1_scores_per_dim.append(scores[2])
    print(f"Macro-F1 {dim}: {scores[2]:.4f}")

overall_macro_f1 = np.mean(f1_scores_per_dim)
print(f"\nOverall Macro-F1 (averaged across dimensions): {overall_macro_f1:.4f}")

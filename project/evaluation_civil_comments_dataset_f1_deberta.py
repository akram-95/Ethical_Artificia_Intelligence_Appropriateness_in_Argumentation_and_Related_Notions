# Evaluation script
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, DebertaForSequenceClassification
## How to run this file: python3 evaluation_civil_comments_dataset_f1_deberta.py
## this file will evaluate all labels under test split in google/civil_comments dataset based on the trained models which saved under: ./appropriateness-classifier
# Defining label columns for Google Civil Comments dataset
label_columns = [
    'toxicity', 'severe_toxicity', 'obscene', 'identity_attack', 'insult',
    'threat'
]

print("Loading dataset...")
dataset = load_dataset("google/civil_comments")

# Taking only 1,500 samples for evaluation
test_dataset = dataset['test'].shuffle(seed=42).select(range(1500))

print("Loading tokenizer and model...")
eval_tokenizer = AutoTokenizer.from_pretrained('./appropriateness-classifier')
eval_model = DebertaForSequenceClassification.from_pretrained('./appropriateness-classifier', ignore_mismatched_sizes=True)

# Function to classify texts
def classify_texts(texts, threshold=0.65):
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
texts = [example['text'] for example in test_dataset]

# Convert ground truth labels to binary (threshold = 0.5)
ground_truth = [[1 if example[label] >= 0.1 else 0 for label in label_columns] for example in test_dataset]

predictions = classify_texts(texts, threshold=0.6)

print("\nCalculating Macro-F1 scores per dimension...")
f1_scores_per_dim = []
for j, dim in enumerate(label_columns):
    true_labels = [x[j] for x in ground_truth]
    pred_labels = [x[j] for x in predictions]
    scores = precision_recall_fscore_support(true_labels, pred_labels, average='macro', zero_division=0)
    f1_scores_per_dim.append(scores[2])
    print(f"Macro-F1 {dim}: {scores[2]:.4f}")

overall_macro_f1 = np.mean(f1_scores_per_dim)
print(f"\nOverall Macro-F1 (averaged across dimensions): {overall_macro_f1:.4f}")

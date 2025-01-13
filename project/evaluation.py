from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
import torch
import numpy as np

# Define dimensions (labels)
DIMS = [
    'Inappropriateness', 'Toxic Emotions', 'Excessive Intensity', 'Emotional Deception',
    'Missing Commitment', 'Missing Seriousness', 'Missing Openness', 'Missing Intelligibility',
    'Unclear Meaning', 'Missing Relevance', 'Confusing Reasoning', 'Other Reasons',
    'Detrimental Orthography', 'Reason Unclassified'
]

# Load the dataset
print("Loading dataset...")
dataset = load_dataset('timonziegenbein/appropriateness-corpus')

# Load the classifier and tokenizer
MODEL_PATH = './appropriateness-classifier'
print("Loading tokenizer and model...")
classifier_tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
classifier_model = BertForSequenceClassification.from_pretrained(MODEL_PATH)


def classify_texts(texts, threshold=0.55):
    """Classify a list of texts and return predictions."""
    all_predictions = []
    print("Classifying texts...")
    for text in texts:
        inputs = classifier_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        with torch.no_grad():
            outputs = classifier_model(**inputs)
            logits = outputs.logits
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
        predictions = (probabilities >= threshold).astype(int)
        all_predictions.append(predictions)
    return all_predictions


# Extract texts and ground truth from the dataset's test split
print("Extracting test data...")
texts = [example['post_text'] for example in dataset['test']]
ground_truth = [[example[label] for label in DIMS] for example in dataset['test']]

# Get predictions for the test texts
predictions = classify_texts(texts, threshold=0.3)

# Calculate Macro-F1 scores for each dimension
print("\nCalculating Macro-F1 scores per dimension...")
f1_scores_per_dim = []
for j, dim in enumerate(DIMS):
    true_labels = [x[j] for x in ground_truth]
    pred_labels = [x[j] for x in predictions]
    scores = precision_recall_fscore_support(true_labels, pred_labels, average='macro')
    f1_scores_per_dim.append(scores[2])  # Append the F1 score for this dimension
    print(f"Macro-F1 {dim}: {scores[2]:.4f}")

# Calculate overall Macro-F1 as the average of individual Macro-F1 scores
overall_macro_f1 = np.mean(f1_scores_per_dim)
print(f"\nOverall Macro-F1 (averaged across dimensions): {overall_macro_f1:.4f}")

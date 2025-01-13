from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
import torch
import numpy as np

# Define the label
LABEL = 'vulgarity'

# Load the dataset
print("Loading dataset...")
dataset = load_dataset('civility-lab/incivility-arizona-daily-star-comments')

# Load the classifier and tokenizer
MODEL_PATH = './vulgarity-classifier'
print("Loading tokenizer and model...")
classifier_tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
classifier_model = BertForSequenceClassification.from_pretrained(MODEL_PATH)


def classify_texts(texts, threshold=0.3):
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
texts = dataset['test']['text']
ground_truth = dataset['test'][LABEL]  # Directly extract the label as a list

# Get predictions for the test texts
predictions = classify_texts(texts, threshold=0.3)

# Convert predictions and ground truth to binary arrays
predictions = [pred[0] if isinstance(pred, np.ndarray) else pred for pred in predictions]

# Calculate precision, recall, and F1 score for the label
print("\nCalculating metrics for vulgarity...")
precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='macro')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Path to the saved model directory
MODEL_PATH = './appropriateness-classifier'

# Load the tokenizer and model from the saved directory
classifier_tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
classifier_model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Initialize the rewriter model (e.g., T5 model for text-to-text generation)
rewriter = pipeline('text2text-generation', model="t5-base")

# Define the inappropriateness prompts for all taxonomy categories
prompt_templates = {
    "toxic emotions": "Please rewrite this text to remove any toxic or overly emotional language: {text}",
    "excessive intensity": "Please rewrite this argument to tone down the intensity: {text}",
    "emotional deception": "Rewrite to remove emotionally manipulative language: {text}",
    "missing commitment": "Rewrite to show stronger commitment: {text}",
    "missing seriousness": "Rewrite to add seriousness: {text}",
    "missing openness": "Rewrite to ensure fairness: {text}",
    "unclear meaning": "Rewrite to improve clarity: {text}",
    "missing relevance": "Rewrite to stay focused on topic: {text}",
    "confusing reasoning": "Rewrite to make reasoning logical: {text}",
    "detrimental orthography": "Rewrite to correct grammar/spelling: {text}",
    "reason unclassified": "Improve this argument for general appropriateness: {text}"
}


def classify_text(text):
    """Classify text into inappropriateness categories."""
    # Tokenize input text
    inputs = classifier_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Perform inference
    with torch.no_grad():
        outputs = classifier_model(**inputs)
        logits = outputs.logits  # Extract logits

    # Apply sigmoid to get probabilities for each label
    probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Define a threshold for each label (usually 0.5 for binary classification)
    predictions = (probabilities >= 0.28).astype(int)  # Convert probabilities to 0 or 1

    # Define label names as per your taxonomy
    label_names = [
        "Inappropriateness", "Toxic Emotions", "Excessive Intensity", "Emotional Deception",
        "Missing Commitment", "Missing Seriousness", "Missing Openness", "Missing Intelligibility",
        "Unclear Meaning", "Missing Relevance", "Confusing Reasoning", "Other Reasons",
        "Detrimental Orthography", "Reason Unclassified"
    ]

    # Combine labels with predictions
    results = {label: int(prediction) for label, prediction in zip(label_names, predictions)}
    return results


def rewrite_text(text, categories):
    """Rewrite text based on detected inappropriateness categories."""
    rewritten_text = text
    for category in categories:
        prompt_template = prompt_templates.get(category)
        if prompt_template:
            prompt = prompt_template.format(text=rewritten_text)
            result = rewriter(prompt, max_length=100, num_return_sequences=1)
            rewritten_text = result[0]['generated_text']
    return rewritten_text


def predict_and_rewrite(text):
    """Predict inappropriateness categories and rewrite the text accordingly."""
    categories = classify_text(text)
    rewritten_text = rewrite_text(text, categories)
    return rewritten_text

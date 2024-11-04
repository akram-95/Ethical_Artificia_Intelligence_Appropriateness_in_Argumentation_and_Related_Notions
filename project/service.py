taxonomy_labels = [
    "Inappropriateness (General)",
    "Toxic Emotions (General)",
    "Excessive Intensity",
    "Emotional Deception",
    "Missing Commitment (General)",
    "Missing Seriousness",
    "Missing Openness",
    "Missing Intelligibility (General)",
    "Unclear Meaning",
    "Missing Relevance",
    "Confusing Reasoning",
    "Other Reasons (General)",
    "Detrimental Orthography",
    "Reason Unclassified"
]
from transformers import pipeline

# Load the model and tokenizer once when the server starts
classifier = pipeline("text-classification", model="./appropriateness-classifier",
                      tokenizer="./appropriateness-classifier")


def classify_text(text):
    # Use the pipeline to get the prediction
    prediction = classifier(text)
    return prediction

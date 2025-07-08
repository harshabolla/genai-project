from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# 1️⃣ Load a pre-trained sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 2️⃣ Example test data with true labels
test_data = [
    {"sentence": "I love this product!", "label": "positive"},
    {"sentence": "This is the worst experience ever.", "label": "negative"},
    {"sentence": "I am so happy with the service.", "label": "positive"},
    {"sentence": "It's terrible and disappointing.", "label": "negative"},
    {"sentence": "Absolutely fantastic!", "label": "positive"},
    {"sentence": "Not what I expected at all.", "label": "negative"}
]

# 3️⃣ Run predictions
sentences = [d["sentence"] for d in test_data]
true_labels = [d["label"] for d in test_data]

predictions = sentiment_pipeline(sentences)
print(f"predictions: {predictions}")

predicted_labels = [pred["label"].lower() for pred in predictions]


# 4️⃣ Evaluate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="binary", pos_label="positive")

print("Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")

# 5️⃣ Print output in required format
print("\nPredictions on Test Data:")
for sentence, pred_label in zip(sentences, predicted_labels):
    print({ "sentence": sentence, "predicted": pred_label })

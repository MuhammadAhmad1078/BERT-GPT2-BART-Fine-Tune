import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def evaluate_model():

    model_path = "./model/bert_sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)


    test_path = "./data/sentiment-analysis.csv"  
    df = pd.read_csv(test_path)
    
    # Use only necessary columns
    df = df[['Text', 'Sentiment']].dropna()

    # Encode true labels
    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    df['label'] = df['Sentiment'].map(label_map)

    texts = df['Text'].tolist()
    true_labels = df['label'].tolist()


    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        outputs = model(**tokens)
        preds = torch.argmax(outputs.logits, dim=1).tolist()


    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='weighted')
    cm = confusion_matrix(true_labels, preds)
    report = classification_report(true_labels, preds, target_names=label_map.keys())

    print("📊 Model Evaluation Results")
    print("=" * 40)
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    evaluate_model()

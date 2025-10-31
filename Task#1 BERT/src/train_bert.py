import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from tokenizer import load_and_tokenize
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import transformers

print("Transformers module path:", transformers.__file__)
from transformers import TrainingArguments
print("TrainingArguments location:", TrainingArguments)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

def train_model():
    print("🚀 Loading datasets and tokenizer...")
    train_dataset, test_dataset, tokenizer = load_and_tokenize()

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    )

    training_args = TrainingArguments(
    output_dir="./model",
    eval_strategy="epoch",         
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=7,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch"
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("Training started...")
    trainer.train()
    results = trainer.evaluate()
    print("📊 Evaluation:", results)

    model.save_pretrained("./model/bert_sentiment")
    tokenizer.save_pretrained("./model/bert_sentiment")

    print("Model and tokenizer saved in ./model/bert_sentiment")

if __name__ == "__main__":
    train_model()

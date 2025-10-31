# ===============================
# BART Fine-Tuning on CNN/DailyMail Dataset
# ===============================

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
import numpy as np
import torch

# ========== Step 1: Load Tokenized Dataset ==========
import pickle

print("Loading pre-tokenized datasets...")
with open('../data/tokenized_datasets.pkl', 'rb') as f:
    tokenized_datasets = pickle.load(f)

# ========== Step 2: Initialize Tokenizer and Model ==========

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)

max_input_length = 512
max_target_length = 150

# ========== Step 4: Load Model ==========

model = BartForConditionalGeneration.from_pretrained(model_name)

# ========== Step 5: Define Metrics ==========

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {k: round(v * 100, 2) for k, v in result.items()}

# ========== Step 6: Training Arguments ==========

training_args = Seq2SeqTrainingArguments(
    output_dir="../model/",
    eval_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=100,
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    logging_dir="../evaluation/logs",
    logging_steps=50,
    predict_with_generate=True,
    generation_max_length=max_target_length,
    generation_num_beams=4
)


# ========== Step 7: Trainer ==========

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# ========== Step 8: Train ==========

trainer.train()

# ========== Step 9: Save Model ==========

trainer.save_model("../model/best_model")
tokenizer.save_pretrained("../model/best_model")

print("Training complete! Model saved in ../model/best_model/")

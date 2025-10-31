import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import BartTokenizer
import pickle

# ========== Step 1: Load Dataset ==========

# Update paths if needed
train_path = "./data/train.csv"
val_path = "./data/validation.csv"
test_path = "./data/test.csv"

# Read the CSVs
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

# Keep only necessary columns
train_df = train_df[['article', 'highlights']].rename(columns={'highlights': 'summary'})
val_df = val_df[['article', 'highlights']].rename(columns={'highlights': 'summary'})
test_df = test_df[['article', 'highlights']].rename(columns={'highlights': 'summary'})

# Create Hugging Face datasets
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df)
})

# ========== Step 2: Initialize Tokenizer ==========

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)

max_input_length = 512
max_target_length = 150

# ========== Step 3: Preprocessing ==========

def preprocess_function(batch):
    # Tokenize input (article)
    inputs = tokenizer(
        batch["article"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )
    
    # Tokenize target (summary) using text_target parameter
    labels = tokenizer(
        text_target=batch["summary"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = labels["input_ids"]
    return inputs

print("Tokenizing datasets... (this may take a few minutes)")
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["article", "summary"])

# Save tokenized datasets
print("Saving tokenized datasets...")
with open('../data/tokenized_datasets.pkl', 'wb') as f:
    pickle.dump(tokenized_datasets, f)

print("Tokenization complete! Datasets saved in ../data/tokenized_datasets.pkl")
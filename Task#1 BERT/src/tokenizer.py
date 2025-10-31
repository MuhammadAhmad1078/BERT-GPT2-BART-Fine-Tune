import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

def load_and_tokenize(model_name="bert-base-uncased"):
    print("Loading preprocessed data...")
    train_df = pd.read_csv('./data/train_preprocessed.csv')
    test_df = pd.read_csv('./data/test_preprocessed.csv')

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["Text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print("Tokenization complete.")
    return train_dataset, test_dataset, tokenizer

if __name__ == "__main__":
    train_dataset, test_dataset, tokenizer = load_and_tokenize()

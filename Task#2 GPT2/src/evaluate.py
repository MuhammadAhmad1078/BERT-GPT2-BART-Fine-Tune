# evaluate.py
# Human-written evaluation script for pseudo-code → code GPT-2 model

import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

print("\n[STEP 1] Loading test data...")
test_path = "../data/pairs.csv"

data = pd.read_csv(test_path)
total_samples = len(data)
print(f"Loaded {total_samples} pseudo→code pairs for evaluation.\n")

# limit to 50 random samples for faster testing
data = data.sample(n=50, random_state=42).reset_index(drop=True)

# ---------------------------------------------------------------
print("[STEP 2] Loading GPT-2 model and tokenizer...")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

checkpoint_path = "../models/gpt2_final.pt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

# handle vocab size mismatch
ckpt_vocab = checkpoint["transformer.wte.weight"].shape[0]
model_vocab = model.transformer.wte.weight.shape[0]
if ckpt_vocab != model_vocab:
    print(f"⚠️ Resizing token embeddings from {model_vocab} → {ckpt_vocab}")
    model.resize_token_embeddings(ckpt_vocab)

# load checkpoint safely
missing, unexpected = model.load_state_dict(checkpoint, strict=False)
print("✅ Checkpoint loaded successfully.")
print("Ignored keys:", missing, unexpected)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ---------------------------------------------------------------
print("\n[STEP 3] Generating code and computing BLEU...\n")

bleu_scores = []
smooth = SmoothingFunction().method1

for idx, row in data.iterrows():
    pseudo = str(row["pseudo_code"])
    target = str(row["python_code"])

    # prepare input + attention mask
    encoded = tokenizer(
        pseudo,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=False
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # manual safe decoding
    token_ids = outputs[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = [t for t in tokens if t is not None]
    generated_text = tokenizer.convert_tokens_to_string(tokens)
    generated_text = generated_text.replace("\n", " ").strip()

    # BLEU score
    bleu = sentence_bleu([target.split()], generated_text.split(), smoothing_function=smooth)
    bleu_scores.append(bleu)

    print(f"\nExample {idx+1}/{len(data)}")
    print("Pseudo-code:")
    print(pseudo)
    print("\nGenerated Code:")
    print(generated_text)
    print("\nActual Code:")
    print(target)
    print(f"BLEU Score: {bleu:.3f}")
    print("-" * 80)

# ---------------------------------------------------------------
avg_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"\n✅ Average BLEU Score on {len(data)} samples: {avg_bleu:.3f}")
print("Evaluation complete.\n")

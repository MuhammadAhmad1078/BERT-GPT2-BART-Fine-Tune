

import os
from transformers import GPT2Tokenizer

base = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(base, "..", "data", "pairs.csv")
save_file = os.path.join(base, "..", "data", "tokenized_data.txt")

# load gpt2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add special markers
tokenizer.add_special_tokens({
    "bos_token": "<START>",
    "eos_token": "<END>",
    "pad_token": "<PAD>"
})
tokenizer.add_tokens(["<CODE>"])

# open csv 
f = open(data_file, "r", encoding="utf-8")
lines = f.readlines()
f.close()

# skip header
new_lines = []
for i in range(1, len(lines)):
    line = lines[i].strip()
    if line == "":
        continue

    # split by first comma 
    comma_index = line.find(",")
    if comma_index == -1:
        continue

    pseudo = line[:comma_index]
    code = line[comma_index + 1:]

    # form training text
    text = "<START> " + pseudo + " <CODE> " + code + " <END>"
    new_lines.append(text)

# tokenize each text manually and save
out = open(save_file, "w", encoding="utf-8")

count = 0
for item in new_lines:
    ids = tokenizer.encode(item)
    ids_str = " ".join(str(x) for x in ids)
    out.write(ids_str + "\n")
    count = count + 1

out.close()

print("tokenized examples written:", count)
print("saved to:", save_file)

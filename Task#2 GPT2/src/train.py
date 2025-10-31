# train.py
# this file will be used in colab to fine tune gpt2
# goal: teach model to turn pseudo steps into python code

import os
import csv
import random
import math
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

# we still need transformers just to get base gpt2 model and tokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# ---------------------------------------------
# 1) read pairs.csv and turn each row into one training string
# ---------------------------------------------

def read_pairs(csv_file_path, limit_rows=None):
    all_rows = []

    f = open(csv_file_path, "r", encoding="utf-8")
    rdr = csv.reader(f)

    # first line is header. we find which index is pseudo_code and which is python_code
    header = next(rdr, None)

    pseudo_i = -1
    code_i = -1

    # detect columns based on names (case-insensitive)
    for idx, col_name in enumerate(header):
        nm = col_name.strip().lower()
        if nm == "pseudo_code":
            pseudo_i = idx
        if nm == "python_code":
            code_i = idx

    if pseudo_i == -1 or code_i == -1:
        raise Exception("could not find pseudo_code/python_code columns in csv header")

    row_counter = 0
    for row in rdr:
        if limit_rows is not None and row_counter >= limit_rows:
            break

        # pull values
        # note: strip just to be safe
        ps = row[pseudo_i].strip()
        py = row[code_i].strip()

        # skip empty lines because dataset sometimes has weird blank trailing rows
        if ps == "" or py == "":
            continue

        # we build one text sample that model will see as one sequence
        # keep it simple and kinda 'chatty' (so viva you can explain easily)
        # important: [PROMPT] and [CODE] are tags we invent so model learns the pattern
        block = ""
        block += "[PROMPT]\n"
        block += ps + "\n\n"
        block += "[CODE]\n"
        block += py + "\n"
        block += "<END>\n"

        all_rows.append(block)
        row_counter += 1

    f.close()
    return all_rows


# ---------------------------------------------
# 2) tiny dataset class for torch
#    (we'll feed GPT-2 causal LM, so labels = input_ids)
# ---------------------------------------------

class MyCodeDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, tokenizer_obj, max_len_tokens):
        self.data_list = text_list
        self.tok = tokenizer_obj
        self.mx = max_len_tokens

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample_text = self.data_list[idx]

        # tokenize one sample
        enc = self.tok(
            sample_text,
            truncation=True,
            max_length=self.mx,
            return_tensors="pt"
        )

        ids = enc["input_ids"].squeeze(0)           # shape: [seq_len]
        att = enc["attention_mask"].squeeze(0)      # shape: [seq_len]

        # labels same as ids (causal LM training)
        # note: we clone so if we ever edit we don't break ids
        lb = ids.clone()

        out = {
            "input_ids": ids,
            "attention_mask": att,
            "labels": lb
        }
        return out


# ---------------------------------------------
# 3) collate_fn for padding batch manually
#    we do this by hand instead of using Trainer
# ---------------------------------------------

def my_collate(batch_list, pad_token_id):
    # figure max length in this batch
    max_len = 0
    for item in batch_list:
        cur_len = item["input_ids"].shape[0]
        if cur_len > max_len:
            max_len = cur_len

    # we will build new padded tensors
    input_batch = []
    mask_batch = []
    label_batch = []

    for item in batch_list:
        ids = item["input_ids"]
        msk = item["attention_mask"]
        lbs = item["labels"]

        need = max_len - ids.shape[0]

        if need > 0:
            # pad input_ids with pad_token_id
            pad_ids = torch.full((need,), pad_token_id, dtype=torch.long)
            ids = torch.cat([ids, pad_ids], dim=0)

            # pad attention_mask with 0 for the padded positions
            pad_mask = torch.zeros(need, dtype=torch.long)
            msk = torch.cat([msk, pad_mask], dim=0)

            # for labels: we want to ignore loss on padded tokens
            # in cross entropy, we usually put -100 to ignore
            pad_lbs = torch.full((need,), -100, dtype=torch.long)
            lbs = torch.cat([lbs, pad_lbs], dim=0)

        input_batch.append(ids.unsqueeze(0))
        mask_batch.append(msk.unsqueeze(0))
        label_batch.append(lbs.unsqueeze(0))

    input_batch = torch.cat(input_batch, dim=0)   # [B, T]
    mask_batch = torch.cat(mask_batch, dim=0)     # [B, T]
    label_batch = torch.cat(label_batch, dim=0)   # [B, T]

    result = {
        "input_ids": input_batch,
        "attention_mask": mask_batch,
        "labels": label_batch
    }
    return result


# ---------------------------------------------
# 4) training loop (manual, basic)
# ---------------------------------------------

def run_train(
    model,
    loader,
    device_used,
    lr_val,
    total_epochs,
    save_folder,
    save_every_steps
):
    # move model to gpu (or cpu if no gpu)
    model.to(device_used)
    model.train()

    # simple AdamW
    opt = optim.AdamW(model.parameters(), lr=lr_val)

    step_no = 0
    loss_log = []

    # create directory if not there
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for ep in range(total_epochs):
        print("starting epoch", ep + 1, "/", total_epochs)

        for batch in loader:
            step_no += 1

            # move tensors to device
            x = batch["input_ids"].to(device_used)
            m = batch["attention_mask"].to(device_used)
            y = batch["labels"].to(device_used)

            # forward pass
            out = model(
                input_ids=x,
                attention_mask=m,
                labels=y
            )
            loss_val = out.loss

            # backward + step
            opt.zero_grad()
            loss_val.backward()
            opt.step()

            # store and print sometimes
            loss_float = float(loss_val.item())
            loss_log.append(loss_float)

            if step_no % 50 == 0:
                print("step", step_no, "loss", loss_float)

            # save checkpoint sometimes so if colab runtime dies you still have progress
            if step_no % save_every_steps == 0:
                ckpt_path = os.path.join(save_folder, "gpt2_step_" + str(step_no) + ".pt")
                torch.save(model.state_dict(), ckpt_path)
                print("saved checkpoint:", ckpt_path)

        print("finished epoch", ep + 1)

    # final save
    final_path = os.path.join(save_folder, "gpt2_final.pt")
    torch.save(model.state_dict(), final_path)
    print("final weights saved at:", final_path)

    return loss_log


# ---------------------------------------------
# 5) main function: glue everything
# ---------------------------------------------

def main():
    # figure base paths
    # this script lives in src/
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "data")
    out_dir = os.path.join(here, "..", "models")

    csv_path = os.path.join(data_dir, "pairs.csv")

    # you can tweak these when you actually run in colab if RAM/GPU explode
    BATCH_SIZE = 4
    MAX_TOKENS = 256
    EPOCHS = 1            # you can increase to 2 or 3 later for final training
    LEARN_RATE = 5e-5
    SAVE_EVERY = 2000     # save checkpoint every N steps
    LIMIT_DEBUG = None    # set small int like 500 if you just want test run

    print("loading csv data from:", csv_path)
    text_samples = read_pairs(csv_path, limit_rows=LIMIT_DEBUG)
    print("total samples:", len(text_samples))

    # shuffle so model doesn't just memorize order of same problem variants
    random.shuffle(text_samples)

    # quick split train/val (90/10). here val is not actually used in loop,
    # but we keep it for future (BLEU etc)
    cut_point = int(len(text_samples) * 0.9)
    train_list = text_samples[:cut_point]
    val_list = text_samples[cut_point:]
    print("train size:", len(train_list), "val size:", len(val_list))

    # load tokenizer + base gpt2 model
    print("loading gpt2 base model/tokenizer ...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    # gpt2 has no pad_token, so we add our own pad token
    # also we tell tokenizer about "<END>" because we used it in our text
    tok.add_special_tokens({"pad_token": "<PAD>"})
    tok.add_special_tokens({"eos_token": "<END>"})

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    # after adding new tokens, we must resize model embedding matrix
    model.resize_token_embeddings(len(tok))

    # wrap into Dataset + DataLoader
    train_ds = MyCodeDataset(train_list, tok, MAX_TOKENS)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: my_collate(b, pad_token_id=tok.pad_token_id)
    )

    # choose device (gpu if available)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("GPU is ON ✔")
    else:
        dev = torch.device("cpu")
        print("GPU not found, using CPU (this will be slow)")

    # run training
    hist = run_train(
        model=model,
        loader=train_loader,
        device_used=dev,
        lr_val=LEARN_RATE,
        total_epochs=EPOCHS,
        save_folder=out_dir,
        save_every_steps=SAVE_EVERY
    )

    # save tokenizer so we can reload later in interface_app
    tok.save_pretrained(out_dir)

    # save the loss numbers in a txt file so we can draw a curve in report
    loss_file_path = os.path.join(out_dir, "loss_values.txt")
    lf = open(loss_file_path, "w", encoding="utf-8")
    for v in hist:
        lf.write(str(v) + "\n")
    lf.close()

    print("training complete. tokenizer and loss log saved in", out_dir)


# ---------------------------------------------
# 6) run main
# ---------------------------------------------

if __name__ == "__main__":
    main()

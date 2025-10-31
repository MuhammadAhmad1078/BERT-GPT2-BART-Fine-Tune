# manual preprocessing of SPOC dataset
# converts basic C++ code style to simple Python style
# and saves aligned pseudo → python pairs in pairs.csv

import os

base_path = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_path, "..", "data", "spoc_raw", "train", "split", "spoc-train-train.tsv")
output_file = os.path.join(base_path, "..", "data", "pairs.csv")

if not os.path.exists(input_file):
    print("❌ File not found:", input_file)
    exit()

# read all lines
with open(input_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

print("✅ total lines read:", len(lines))
print("🔍 first 3 sample lines:\n")
for l in lines[:3]:
    print(l.strip())

# create output csv
out = open(output_file, "w", encoding="utf-8")
out.write("pseudo_code,python_code\n")

count = 0

# skip the first header line
for i in range(1, len(lines)):
    line = lines[i].strip()
    if line == "":
        continue

    parts = line.split("\t")
    if len(parts) < 2:
        continue

    pseudo = parts[0].strip()
    code_cpp = parts[1].strip()

    # --- manual C++ → Python cleanup ---
    code = code_cpp
    code = code.replace("\\n", "\n")
    code = code.replace("\t", " ")
    code = code.replace("{", "")
    code = code.replace("}", "")
    code = code.replace(";", "")
    code = code.replace("cout <<", "print(")
    code = code.replace("<< endl", ")")
    code = code.replace("<<", ",")
    code = code.replace("cin >>", "input() #")
    code = code.replace("std::", "")
    code = code.replace("using namespace std", "")
    code = code.replace("#include", "#")
    code = code.replace("main()", "main():")
    code = code.replace("for(", "for ")
    code = code.replace("if(", "if ")
    code = code.replace("else if", "elif")
    code = code.replace("else{", "else:")
    code = code.replace("printf", "print")
    code = code.replace("//", "#")
    code = code.replace("return 0", "")
    code = code.replace("int ", "")
    code = code.replace("float ", "")
    code = code.replace("double ", "")
    code = code.replace("char ", "")
    code = code.replace("bool ", "")
    code = code.replace("true", "True")
    code = code.replace("false", "False")

    # cleanup formatting
    lines_clean = []
    for c in code.split("\n"):
        c = c.strip()
        if c != "" and c not in ["{", "}"]:
            lines_clean.append(c)
    code_final = " ".join(lines_clean)

    # remove commas for safe CSV
    pseudo = pseudo.replace(",", " ")
    code_final = code_final.replace(",", " ")

    out.write(pseudo + "," + code_final + "\n")
    count += 1

out.close()

print("\nConverted C++ → Python-like pairs saved in:", output_file)
print(" Total pairs written:", count)
print(" Check your data/pairs.csv file to confirm — now it should contain pseudo → code lines.")

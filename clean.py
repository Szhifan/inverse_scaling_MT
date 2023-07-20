import re
import os
ref_paths = ["datasets/truthfullqa/ref_en.txt", "datasets/truthfullqa/ref_de.txt","datasets/truthfullqa/ref_fr.txt"]
bad_sentences = set()

for path in ref_paths:
    text = open(path,"r").readlines()
    for i,sent in enumerate(text):
        if re.search(r"\.\n$",sent):
            bad_sentences.add(i)


def clean(path):
    text = open(path,"r").readlines()
    new_text = [sent for i,sent in enumerate(text) if i not in bad_sentences]
    with open(path,"w") as f:
        f.writelines(new_text)


root = "datasets/truthfullqa/"
for i in os.listdir(root):
    if i.endswith("output"):
        for f in os.listdir(root+i):
            path = root+i+"/"+f
            
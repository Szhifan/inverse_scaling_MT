from datasets import load_dataset
import models
import tqdm 
import pandas as pd 
import pickle 
import re
questions_src = load_dataset("truthful_qa","generation")["validation"]["question"]
translator = models.get_model("mbart-large-50-one-to-many-mmt","English","Russian",False)
questions_tgt= []
for s in tqdm.tqdm(questions_src):
    tgt = translator(s)
    if not re.search(r"\.$",tgt): 
        questions_tgt.append(tgt)

parallel = list(zip(questions_src,questions_tgt))
df = pd.DataFrame(parallel)
df.columns = ["en","fr"]
out_dir = "datasets/truthfullqa/en_ro.df"
pickle.dump(df,open(out_dir,"wb"))

ref_dir = "datasets/truthfullqa/ref_ru.txt"
with open(ref_dir,"a") as f:
    for sent in questions_tgt:
        f.write(sent)
        f.write("\n")
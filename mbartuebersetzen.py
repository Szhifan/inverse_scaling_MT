from datasets import load_dataset
import models
import tqdm 
import pandas as pd 
import pickle 
questions_src = load_dataset("truthful_qa","generation")["validation"]["question"]

translator = models.get_model("mbart-large-50-one-to-many-mmt","English","German")

questions_tgt= []
for s in tqdm.tqdm(questions_src):
    tgt = translator(s)
    questions_tgt.append(tgt)

parallel = list(zip(questions_src,questions_tgt))
df = pd.DataFrame(parallel)
df.columns = ["en","de"]
out_dir = "datasets/truthfullqa/en_de.df"
pickle.dump(df,open(out_dir,"wb"))

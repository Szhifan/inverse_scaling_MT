from transformers import pipeline
from utils import eval 
from load_data import *
import os 
import load_data 
import pickle 
ref_dir = "datasets/truthfullqa/ref_ro.txt"
df_dir = "datasets/truthfullqa/en_ro.df"

df = pickle.load(open(df_dir,"rb"))
with open(ref_dir,"a") as f:
    for sent in df["ro"]:
        f.write(sent)
        f.write("\n")



    
    
    
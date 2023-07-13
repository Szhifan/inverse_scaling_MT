from transformers import pipeline
from utils import eval 
from load_data import *
import os 
import load_data 
import pickle 
ref_dir = "datasets/truthfullqa/ref_ro.txt"
df_dir = "datasets/truthfullqa/en_ro.df"

df = pickle.load(open(df_dir,"rb"))
df.columns = ["en","ro"]
pickle.dump(df,open(df_dir,"wb"))


    
    
    
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from datasets import load_dataset,Dataset,disable_caching
import pandas as pd 
import pickle 

class Parallel_dataset:
    def __init__(self,dir:str) -> None:
        #dir is a dataframe file. 
        self.df = pickle.load(open(dir,"rb"))
        self.data = Dataset.from_pandas(self.df)
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return self.data[index]

    
    

if __name__ == "__main__":      
    dts = Parallel_dataset("datasets/europarl/ro-en/ro_en.df")
    for i in dts:
        print(i)
    
    
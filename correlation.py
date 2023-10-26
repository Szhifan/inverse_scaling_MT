from scipy.stats import pearsonr
import os
from utils import extract_stats
root_dir = "results/"
import re 
out_path = "results/llama2_pearson.txt"
regex = r"_([a-z]{2}_[a-z]{2})"
for dir in os.listdir(root_dir):
    model_name = dir
    dir = root_dir + dir
    if "Llama" not in dir:
        continue
    for dir2 in os.listdir(dir):
        dir2 = dir + "/" + dir2
        if "prefix" in dir2:
            prefix_model_name = model_name + "-prefix"
            for dir3 in os.listdir(dir2):
         
                lang_pair = re.search(regex,dir3).group(1)
                string = prefix_model_name + "-" + lang_pair
                dir3 = dir2 + "/" + dir3
            
                stats = extract_stats(dir3)
                res = pearsonr(stats["size"],stats["accuracy"])
                f = open(out_path,'a+')
                f.write(f"{string}: {res}\n")
        else:
            lang_pair = re.search(regex,dir2).group(1)
            string = model_name + "-" + lang_pair
            
            stats = extract_stats(dir2)
            res = pearsonr(stats["size"],stats["accuracy"])
            f = open(out_path,'a+')
            f.write(f"{string}: {res}\n")

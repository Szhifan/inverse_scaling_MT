from matplotlib import pyplot as pp 
from utils import *
import re
import numpy as np 


log_dir_de = "results/t5/thruthfullqa_en_de.txt" #modify file name!
def parse_log(log_dir):
    info = extract_stats(log_dir)
    data = {"model_name":[],"accuracy":[],"num_params":[],"bleu":[]}
    for item in info["data"]:
        data["model_name"].append(item["model_name"])
        data["accuracy"].append(item["accuracy"])
        data["bleu"].append(item["bleu"])
        data["num_params"].append(item["num_params"])
    return data 
de = parse_log("results/t5/thruthfullqa_en_de.txt")
fr = parse_log("results/t5/thruthfullqa_en_fr.txt")
ro = parse_log("results/t5/thruthfullqa_en_ro.txt")

def plot(metric:str):


    pp.xticks([10**i for i in range(8,12)])
    pp.xscale("log")
    pp.plot("num_params",metric,data=de,marker="+",color="r",label="German") ##
    pp.plot("num_params",metric,data=fr,marker="o",color="b",label="French") ##
    pp.plot("num_params",metric,data=fr,marker="1",color="y",label="Romanian") ##
    pp.legend()
    pp.xlabel("model size")
    pp.ylabel(metric) ###
    pp.title(f"{id2lang[src_id]}-{id2lang[tgt_id]}:plot of model size vs {metric}") #modify figure path!
    pp.savefig(f"figures/t5/{metric}_{src_id}_{tgt_id}.jpg")  ###
plot("accuracy")
plot("bleu")
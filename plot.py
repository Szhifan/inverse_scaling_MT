from matplotlib import pyplot as pp 
from utils import *
import re
import numpy as np 


log = "results/openai/thruthfullqa_de_en.txt" #modify file name!

stats = extract_stats(log)

def parse_info(info:dict):
    instruct_gpt = {"num_params":[],"accuracy":[],"bleu":[]}
    gpt3 = {"num_params":[],"accuracy":[],"bleu":[]}
    gpt35 = {"num_params":[],"accuracy":[],"bleu":[]}
    for item in info:
        model_name = item["model_name"]
        if model_name.endswith("001"):

            instruct_gpt["num_params"].append(item["num_params"])
            instruct_gpt["accuracy"].append(item["accuracy"])
            instruct_gpt["bleu"].append(item["bleu"])
        elif model_name.endswith("002") or model_name.endswith("003"):
            gpt35["num_params"].append(item["num_params"])
            gpt35["accuracy"].append(item["accuracy"])
            gpt35["bleu"].append(item["bleu"]) 

        else:
            gpt3["num_params"].append(item["num_params"])
            gpt3["accuracy"].append(item["accuracy"])
            gpt3["bleu"].append(item["bleu"])

    return instruct_gpt,gpt3,gpt35


insgpt,gpt3,_=parse_info(stats)  
# print(insgpt)   
def plot(metric:str,src_id,tgt_id):
    pp.xticks([10**i for i in range(8,12)])
    pp.xscale("log")
    pp.plot("num_params",metric,data=insgpt,marker="+",color="r",label="instruct gpt") ##
    pp.plot("num_params",metric,data=gpt3,marker="o",color="b",label="gpt 3") ##

    pp.legend()
    pp.xlabel("model size")
    pp.ylabel(metric) ###
    pp.title(f"{id2lang[src_id]}-{id2lang[tgt_id]}:plot of model size vs {metric}") #modify figure path!
    pp.savefig(f"figures/openai/{metric}_{src_id}_{tgt_id}.jpg")  ###

plot("accuracy","de","en")
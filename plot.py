from matplotlib import pyplot as pp 
from utils import *
import re
import numpy as np 
from scipy.stats import pearsonr
import numpy as np

log = "results/flan-t5/thruthfullqa_en_ro.txt" #modify file name!

stats = extract_stats(log)

def parse_info(info:dict):
    instruct_gpt = {"size":[],"accuracy":[],"bleu":[]}
    gpt3 = {"size":[],"accuracy":[],"bleu":[]}
    text_dvnc_2 = {"size":[],"accuracy":[],"bleu":[]}
    text_dvnc_3 = {"size":[],"accuracy":[],"bleu":[]}
    for item in info:
        model_name = item["model_name"]
        if model_name.endswith("001"):

            instruct_gpt["size"].append(item["size"])
            instruct_gpt["accuracy"].append(item["accuracy"])
            instruct_gpt["bleu"].append(item["bleu"])
        elif model_name.endswith("002"):
            text_dvnc_2["size"].append(item["size"])
            text_dvnc_2["accuracy"].append(item["accuracy"])
            text_dvnc_2["bleu"].append(item["bleu"]) 
        elif model_name.endswith("003"):

            text_dvnc_3["size"].append(item["size"])
            text_dvnc_3["accuracy"].append(item["accuracy"])
            text_dvnc_3["bleu"].append(item["bleu"]) 

        else:
            gpt3["size"].append(item["size"])
            gpt3["accuracy"].append(item["accuracy"])
            gpt3["bleu"].append(item["bleu"])

    return instruct_gpt,gpt3,text_dvnc_2,text_dvnc_3


def parse_info_t5(info:dict):
    t5 = {"size":[],"accuracy":[],"bleu":[]}
    flan_t5 = {"size":[],"accuracy":[],"bleu":[]}
    for item in info:
        model_name = item["model_name"]
        if model_name.startswith("t5"):
            t5["size"].append(item["size"])
            t5["accuracy"].append(item["accuracy"])
            t5["bleu"].append(item["bleu"])
        elif model_name.startswith("flan"):
            flan_t5["size"].append(item["size"])
            flan_t5["accuracy"].append(item["accuracy"])
            flan_t5["bleu"].append(item["bleu"])
    return t5,flan_t5


  
def plot(metric:str,src_id,tgt_id):
    log = f"results/openai/thruthfullqa_{src_id}_{tgt_id}.txt"
    stats = extract_stats(log)
    insgpt,gpt3,text_dvnc_2,text_dvnc_3=parse_info(stats)

    pp.xticks([10**i for i in range(8,12)])
    pp.xscale("log")
    pp.plot("size",metric,data=insgpt,marker="+",color="r",label="instruct gpt") ##
    pp.plot("size",metric,data=gpt3,marker="o",color="b",label="gpt 3") ##
    pp.plot("size",metric,data=text_dvnc_2,marker="x",color="y",label="text-davinci-002") ##
    pp.plot("size",metric,data=text_dvnc_3,marker="v",color="k",label="text-davinci-003") ##
    pp.legend()
    pp.xlabel("model size")
    pp.ylabel(metric) ###
    pp.title(f"{id2lang[src_id]}-{id2lang[tgt_id]}:plot of model size vs {metric}") #modify figure path!
    pp.savefig(f"figures/openai/{metric}_{src_id}_{tgt_id}.jpg")  ###

outputs  = parse_info_t5(stats)

def pearson(info:dict):
    acc = np.array(info["accuracy"]) 
    size = np.array(info["size"]) / 1000000 
    if len(acc)>=2:
        return pearsonr(size,acc)
  

for output in outputs:
    print(pearson(output))

from matplotlib import pyplot as pp 
from utils import *
import re
import numpy as np 


log_dir = "results/t5/thruthfullqa_en_de.txt" #modify file name!
info = extract_stats(log_dir)
gpt_3 = {"model_name":[],"accuracy":[],"num_params":[],"bleu":[]}
instruct_gpt = {"model_name":[],"accuracy":[],"num_params":[],"bleu":[]}
gpt_35 = {"model_name":[],"accuracy":[],"num_params":[],"bleu":[]}
for item in info["data"]:
    if not item["model_name"].startswith("text") and not item["model_name"].startswith("gpt"):
        gpt_3["model_name"].append(item["model_name"])
        gpt_3["accuracy"].append(item["accuracy"])
        gpt_3["bleu"].append(item["bleu"])
        gpt_3["num_params"].append(item["num_params"])

    elif re.search(r"text\-[a-z]+\-001",item["model_name"]):

        instruct_gpt["model_name"].append(item["model_name"])
        instruct_gpt["accuracy"].append(item["accuracy"])
        instruct_gpt["bleu"].append(item["bleu"])
        instruct_gpt["num_params"].append(item["num_params"])
    else:
        gpt_35["model_name"].append(item["model_name"])
        gpt_35["accuracy"].append(item["accuracy"])
        gpt_35["bleu"].append(item["bleu"])
        gpt_35["num_params"].append(item["num_params"])
instruct_gpt["bleu"].sort()
instruct_gpt["accuracy"].sort()
instruct_gpt["num_params"].sort()
def plot(metric:str):
    src_id = re.search(r"([a-z]{2})_([a-z]{2})\.txt",log_dir).group(1)
    tgt_id = re.search(r"([a-z]{2})_([a-z]{2})\.txt",log_dir).group(2)

    pp.xticks([10**i for i in range(8,12)])
    pp.xscale("log")
    pp.plot("num_params",metric,data=instruct_gpt,marker="+",color="r",label="instruct gpt") ##
    pp.plot("num_params",metric,data=gpt_3,marker="o",color="b",label="gpt3") ##
    pp.legend()
    pp.xlabel("model size")
    pp.ylabel(metric) ###
    pp.title(f"{id2lang[src_id]}-{id2lang[tgt_id]}:plot of model size vs bleu score") #modify figure path!
    pp.savefig(f"figures/t5/{metric}_{src_id}_{tgt_id}.jpg")  ###
plot("accuracy")
plot("bleu")
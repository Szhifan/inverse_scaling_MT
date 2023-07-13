import argparse 
from models import * 
from utils import *
from load_data import Parallel_dataset
import re 
import tqdm
import os 
from transformers import pipeline

def add_training_args(parser):
    parser.add_argument("--model-name",type=str,help="choose a model.")
    parser.add_argument("--dataset",type=str,help="choose a dataset")
    parser.add_argument("--log-file",type=str,default=None,help="path to save the log")
    parser.add_argument("--few-shot",action="store_true",help="specify if few shot prompt is needed.")
    parser.add_argument("--lang-pair",type=str,help="indicating the language pair, the first one is the source language and the second one is the target language.")

def get_args():
    parser = argparse.ArgumentParser()
    add_training_args(parser)
    args = parser.parse_args()
    return args 


def main(args): 
    init_logging(args)
    logging.info("start experiment...")
    src_id,tgt_id = args.lang_pair.split("-")[0],args.lang_pair.split("-")[1]
    src_lang,tgt_lang = id2lang[src_id],id2lang[tgt_id] 

    logging.info(f"language pair: {src_lang}-{tgt_lang}")
    
    #load model and dataset 
    model = get_model(args.model_name,src_lang,tgt_lang,few_shot=True if args.few_shot else False)
    logging.info(f"model parameters: {model.num_params}")
    data_set = Parallel_dataset(args.dataset)
    translation_output_dir = re.sub(r"[a-z]{2}_[a-z]{2}\.df",f"{src_id}_{tgt_id}_output",args.dataset)+f"/{model.model_name}.txt"
    os.makedirs(os.path.dirname(translation_output_dir),exist_ok=True)
    f = open(translation_output_dir,"a")
    for pair in tqdm.tqdm(data_set.to_list()):

  
        data = {"src":pair[src_id],"mt":model(pair[src_id]),"ref":pair[tgt_id]} 
        f.write(data["mt"]+"\n")
    f.close()
    ref_dir = f"datasets/truthfullqa/ref_{tgt_id}.txt"

    stats = eval(ref_dir,translation_output_dir)
    os.system("rm ./ref.txt")
    logging.info(stats)
    logging.info("="*20)
      


if __name__ == "__main__":
    args = get_args()
    main(args)

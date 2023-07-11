import argparse 
from models import * 
from utils import *
from load_data import Parallel_dataset
import re 
import tqdm
import os
import evaluate 

def add_training_args(parser):
    parser.add_argument("--model-name",type=str,help="choose a model.")
    parser.add_argument("--dataset",type=str,help="choose a dataset")
    parser.add_argument("--log-file",type=str,default=None,help="path to save the log")
    parser.add_argument("--few-shot",type=bool,default=True,help="specify if few shot prompt is needed.")
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
    model = get_model(args.model_name,src_lang,tgt_lang)
    logging.info(f"model parameters: {model.num_params}")
    data_set = Parallel_dataset(args.dataset)
    translation_output_dir = re.sub(r"[a-z]{2}_[a-z]{2}\.df",f"{args.lang_pair}_output",args.dataset)+f"/{model.model_name}.txt"
    os.makedirs(os.path.dirname(translation_output_dir),exist_ok=True)

    f = open(translation_output_dir,"a")
    for pair in tqdm.tqdm(data_set):
        data = {"src":pair[src_id],"mt":model(pair[src_id]),"ref":pair[tgt_id]}
        f.write(data["mt"]+"\n")
    f.close()
    tgt_sents = list(data_set.df[tgt_id]) 
    ref_dir = "ref.txt"
    with open(ref_dir,"a") as f:
        for sent in tgt_sents:
            f.write(sent)
            f.write("\n")
    
    #calculating bleu score
    os.system(f"./bleu.sh {ref_dir} {translation_output_dir} {args.log_file}")
    logging.info("="*20)
      


if __name__ == "__main__":
    args = get_args()
    main(args)

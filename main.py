import argparse 
from models import * 
from utils import *
from load_data import Parallel_dataset
import re 
import tqdm
import evaluate 
id2lang = {"en":"English","ro":"Romanian","fr":"French","de":"German"}
def add_training_args(parser):
    parser.add_argument("--model-name",type=str,help="choose a model.")
    parser.add_argument("--dataset",type=str,help="choose a dataset")
    parser.add_argument("--log-file",type=str,default=None,help="path to save the log")
    parser.add_argument("--few-shot",type=bool,default=True,help="specify if few shot prompt is needed.")

def get_args():
    parser = argparse.ArgumentParser()
    add_training_args(parser)
    args = parser.parse_args()
    return args 


def main(args): 
    init_logging(args)
    logging.info("start experiment...")
    lang_pair = re.search(r"([a-z]{2})_([a-z]{2})\.df",args.dataset) 
    src_id = lang_pair.group(1)
    tgt_id = lang_pair.group(2)
    src_lang,tgt_lang = id2lang[src_id],id2lang[tgt_id] 

    logging.info(f"language pair: {src_lang}-{tgt_lang}")
    
    #load model and dataset 
    model = get_model(args.model_name,src_lang,tgt_lang)
    logging.info(f"model parameters: {model.num_params}")
    data_set = Parallel_dataset(args.dataset)

    #load evaluattion metrics 
    bleu = evaluate.load("bleu")
  

    bleu_score = 0 

    translation_output_dir = re.sub(r"\.df",r"_output",args.dataset)+f"/{model.model_name}.txt"
    print(translation_output_dir)
    f = open(translation_output_dir,"a")

    for pair in tqdm.tqdm(data_set):
        data = {"src":pair[src_id],"mt":model(pair[src_id]),"ref":pair[tgt_id]}
        try: 
            bleu_score += bleu.compute(predictions=[data["mt"]],references=[data["ref"]])["bleu"]
        except:
            pass 
        f.write(data["mt"]+"\n")
        
    bleu_score = bleu_score/len(data_set)
  

    logging.info(f"bleu score: {bleu_score}")
    logging.info("="*20)
      


if __name__ == "__main__":
    args = get_args()
    main(args)

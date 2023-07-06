import argparse 
from models import * 
from utils import *
from load_data import Parallel_dataset
import re 
import tqdm
import evaluate 
from comet import load_from_checkpoint,download_model
id2lang = {"en":"English","ro":"Romanian","fr":"French","de":"German"}
def add_training_args(parser):
    parser.add_argument("--model-name",type=str,help="choose a model.")
    parser.add_argument("--dataset",type=str,help="choose a dataset")
    parser.add_argument("--log-file",default=None,help="path to save the log")

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
    comet_model_path = download_model("Unbabel/wmt22-comet-da")

    comet_model = load_from_checkpoint(comet_model_path)    

    bleu_score = 0 
    comet_score = 0 

    for pair in tqdm.tqdm(data_set):
        
        data = [{"src":pair[src_id],"mt":model(pair[src_id]),"ref":pair[tgt_id]}]
  
        bleu_score += bleu.compute(predictions=[data["mt"]],references=[data["ref"]])["bleu"]
        comet_score += comet_model.predict(data)[0]
    bleu_score = bleu_score/len(data_set)
    comet_model = comet_model/len(data_set)

    logging.info(f"bleu score: {bleu_score} comet score: {comet_score}")
        
    #the dataset should be a data frame with two columns: 
    #src lang and tgt lang, one sentence per row. 
    # each hypothesis translation should goes to a new column of the data frame. 
    # after translating the entire data frame, comet and blue scores are computed for each 
    # experiment. 

if __name__ == "__main__":
    args = get_args()
    main(args)

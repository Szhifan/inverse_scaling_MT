import logging 
import os
import sys 
import math 
from transformers import pipeline
import re
import tqdm 
id2lang = {"en":"English","ro":"Romanian","fr":"French","de":"German"}
lang2id = {kp[1]:kp[0] for kp in id2lang.items()}
def init_logging(args):
    handlers = [logging.StreamHandler()]
    if hasattr(args, 'log_file') and args.log_file is not None:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, mode='a+'))
    logging.basicConfig(handlers=handlers, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.info('COMMAND: %s' % ' '.join(sys.argv))
    logging.info('Arguments: {}'.format(vars(args)))

def eval(ref_dir,mt_dir) -> str:
    """
    evaluates the translation quality. It contains three metrics:
    1. bleu score 
    2. question mark accuracy: if the sentence ends with question marks
    3. language id accuracy: if the sentence is the same language as the target language 
    Note that 2. and 3. apply only to truthfullqa dataset 

    return: str
    """
    os.system("touch ./bleu.txt")
    os.system(f"perl multi-bleu.perl -lc {ref_dir} < {mt_dir} >> ./bleu.txt")
    bleu_score = open("bleu.txt","r").read()
    bleu_score = re.search(r"BLEU = ([0-9]+\.[0-9]+),",bleu_score).group(1)
    os.system("rm ./bleu.txt")
    
    stats = f"bleu score: {bleu_score}"
    if re.search("datasets/truthfullqa",mt_dir):
        f = open(mt_dir,"r")
        mt_text = f.readlines()
        n = len(mt_text)
        q_acc = 0 
        l_acc = 0 
        for sent in tqdm.tqdm(mt_text):
            sent = sent[:-1]
            sent = re.sub(f"\"","",sent)
            if sent.endswith("?"):
                q_acc += 1 
            # lang_id = lang_id_ppl(sent)[0]["label"]
            # if lang_id == tgt_id:
            #     l_acc += 1
        f.close()
        l_acc = l_acc / n
        q_acc = q_acc / n
 
        stats = stats + f"|language acc:{l_acc}|question mark acc:{q_acc}"
    return stats

def extract_stats(path:str):
    file = open(path,"r").read()
    lang_pair = re.search(r"language pair:\s([a-zA-Z]+\-[a-zA-Z]+)\n",file).group(1)
    stats = {"language pair":lang_pair,"data":[]}
    ex_items = re.split("====================",file)
    for ex in ex_items:
        print(ex)
        item = {}
        item["model_name"] = re.search(r"--model-name\s(.+?)\s",ex).group(1)
        item["num_params"] = float(re.search(r"model parameters:\s(.+?)\n",ex).group(1))
        item["bleu"] = float(re.search(r"bleu score:\s([0-9]+\.[0-9]+)",ex).group(1))
        item["accuracy"] = round(float(re.search(r"question mark acc:([0-9]+\.[0-9]+)",ex).group(1)),4) 
        stats["data"].append(item)
    
    return stats



    


if __name__ == "__main__":
    print(extract_stats("results/t5/thruthfullqa_en_de.txt"))
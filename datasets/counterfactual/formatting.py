import json 
import re 
dir = "datasets/counterfactual/counterfact.json"
def js_to_df(dir):
    js = json.load(open(dir,"r"))
    for item in js:
    
        
        prompt = item["requested_rewrite"]["prompt"]
        subject = item["requested_rewrite"]["subject"]
        target_new = item["requested_rewrite"]["target_new"]["str"]
        counterfact_prompt = re.sub(r"\{\}",subject,prompt) + f" {target_new}"
        
js_to_df(dir)


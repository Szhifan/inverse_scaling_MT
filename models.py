import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig  # type: ignore
from transformers import AutoModelForSeq2SeqLM
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import re
import openai 
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from typing_extensions import Literal, get_args
import torch 
import transformers 

device = "cpu"
# DEBUG: counting errors
error_count = 0

#Pre-define some valid Hugging Face models. 
ValidHFModel = [
    "mbart-large-50-many-to-one-mmt",
    "mbart-large-50-one-to-many-mmt",
    "flan-t5-small",
    "flan-t5-base",
    "flan-t5-large",
    "flan-t5-xl",
    "flan-t5-xxl",
    "mt5-base",
    "mt5-large",
    "mt5-small",
    "mt5-medium",
    "mt5-xxl"

]




mbart_lang_ids = {"English":"en_XX","German":"de_DE","Russian":"ru_RU"}

class HFTranslator():
    def __init__(self,model_name:ValidHFModel,src_lang,tgt_lang) -> None:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang 
        self.model_name = model_name
        self.need_prompt = True
        use_fast = True 
        if model_name.startswith("flan-") or model_name.startswith("mt5"):
            if (model_name.startswith("flan-")) and (src_lang not in ["German","English","Romanian","French"]) and (tgt_lang not in ["German","English","Romanian","French"]):
                raise ValueError("language not supported by flan-t5!")
            prefix = "google/"
            torch.cuda.empty_cache()
            self.model = AutoModelForSeq2SeqLM.from_pretrained(prefix + model_name, max_length=1024).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(
            prefix + model_name,
            use_fast=use_fast,
            model_max_length=1023
        )
        if model_name.startswith("mbart"):
            self.need_prompt = False
            src_lang_id = mbart_lang_ids[self.src_lang]
            prefix = "facebook/"  
            self.model = MBartForConditionalGeneration.from_pretrained(prefix+model_name).to(device)
            self.tokenizer = MBart50TokenizerFast.from_pretrained(prefix+model_name)
            self.tokenizer.src_lang = mbart_lang_ids[self.src_lang]
      
        self.num_params = sum(p.numel() for p in self.model.parameters())
  
    def get_prompt(self,text,format=2):
        """
        Construct MT prompt format: 
        translate [src] to [tgt]: text 
        """
        src = self.src_lang
        tgt = self.tgt_lang 
        prompt = f"[{src}]: {text} \n[{tgt}]:" if format==1 else f"translate {src} to {tgt}: {text}"
        return prompt 
 
    
    def __call__(self,src_text:str):
        if self.need_prompt:
            prompt = self.get_prompt(src_text)
            print(prompt)
            input_ids = self.tokenizer(prompt,return_tensors="pt")["input_ids"].to(device)
            outputs = self.model.generate(input_ids)
        else:
            input_ids = self.tokenizer(src_text,return_tensors="pt")["input_ids"].to(device)
            if re.search(r"-to-many",self.model_name):
                outputs = self.model.generate(input_ids,forced_bos_token_id=self.tokenizer.lang_code_to_id[mbart_lang_ids[self.tgt_lang]])
            else:
                outputs = self.model.generate(input_ids)
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation

ValidOpenAiModel =[ 
    "gpt-4",
    "gpt-3.5-turbo",
    "text-davinci-001",
    "text-davinci-002",
    "text-davinci-003"
]

class OpenAiTranslator():
    def __init__(self,model_name:ValidOpenAiModel,src_lang,tgt_lang) -> None:
        openai.api_key = "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"
        self.model_name = model_name
        self.src_lang = src_lang 
        self.tgt_lang = tgt_lang 
    def get_prompt(self,text):
        """
        Construct MT prompt format: 
        translate [src] to [tgt]: text 
        """
        prompt = f"translate {self.src_lang} to {self.tgt_lang}: \"{text}\""
        return prompt 
    def extract_ans(self,response):
        reg_format_1 = r"translates to \"(.+)\" in"
        reg_format_2 = r"\"(.+)\" is the translation of"
        reg_format_3 = r"I am sorry .+ offensive language"
        reg_format_4 = r"^translation: \"(.+)\"$"
        reg_formate_5 = r"^\"(.+)\"$"
        if re.search(reg_format_1,response):
            translation = re.search(reg_format_1,response).group(1)
        elif re.search(reg_format_2,response):
            translation = re.search(reg_format_2,response).group(1)
        elif re.search(reg_format_3,response):
            return None
        elif re.search(reg_format_4,response):
            translation = re.search(reg_format_4,response).group(1)
        elif re.search(reg_formate_5,response):
            translation = re.search(reg_formate_5).group(1)
        return translation
    def _call_api(self,text,reference=None):
        prompt = self.get_prompt(text)
        response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0, 
            )['choices'][0]['message']['content']
        return response
        


        
    def __call__(self,text,reference=None):
        response = self._call_api(text)
  
def get_model(model_name,src_lang,tgt_lang):
    if model_name in ValidHFModel:
        model = HFTranslator(model_name,src_lang,tgt_lang)
    elif model_name in ValidOpenAiModel:
        model = OpenAiTranslator(model_name,src_lang,tgt_lang)
    else:
        raise ValueError("please enter a valid model name!")
    return model 


        


if __name__ == "__main__":
    text = "I can see you."
    translator = HFTranslator("flan-t5-base","English","Romanian")
    print(translator(text))

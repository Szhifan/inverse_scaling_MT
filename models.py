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
from nltk import sent_tokenize 

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


mbart_lang_ids = {"English":"en_XX","German":"de_DE","Russian":"ru_RU","French":"fr_XX","Romanian":"ro_RO"}



class HFTranslator():
    def __init__(self,model_name:ValidHFModel,src_lang,tgt_lang,few_shot=False) -> None:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang 
        self.model_name = model_name
        self.need_prompt = True
        self.few_shot = few_shot
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
  
    def get_prompt(self,text):
        """
        Construct MT prompt format: 
        translate [src] to [tgt]: text 
        """
        src = self.src_lang
        tgt = self.tgt_lang 
        prompt = f"[{src}]: {text} \n[{tgt}]:"
        return prompt 
 
    
    def __call__(self,src_text:str):
        if self.need_prompt:
            prompt = self.get_prompt(src_text)
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

ValidOpenAiModel ={
    "gpt-3.5-turbo":"175B",
    "text-davinci-001":"175B",
    "text-davinci-002":"175B",
    "text-davinci-003":"175B",
    "text-curie-001":"6.7B",
    "text-babbage-001":"175B",
    "text-ada-001":"350M",
    "davinci":"175B",
    "curie":"6.7B",
    "babbage":"1.3B",
    "ada":"350M"
}
few_shot_example = "[English]: I love you.\n[German]: Ich liebe dich."
class OpenAiTranslator():
    def __init__(self,model_name:ValidOpenAiModel,src_lang,tgt_lang,few_shot=True) -> None:
        openai.api_key = "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"
        self.model_name = model_name
        self.src_lang = src_lang 
        self.tgt_lang = tgt_lang
        self.num_params = ValidOpenAiModel[self.model_name]
        self.few_shot = few_shot # determine if few-shot-prmopt is needed. 
    def get_prompt(self,src_text):
        """
        Construct MT prompt format: 
        translate [src] to [tgt]: text 
        """
        prompt = f"translate {self.src_lang} to {self.tgt_lang}: \"{src_text}\""
        if self.few_shot: 
            prompt = few_shot_example + f"\n[{self.src_lang}]:{src_text}\n[{self.tgt_lang}]:"
        return prompt
    def extract_ans(self,response):
        translation = re.sub(r"\n","",response)
        if self.few_shot: 
            try:
                translation = sent_tokenize(response)[0] 
            except: 
                pass 
        return translation
    def _call_api(self,text):
        prompt = self.get_prompt(text)
        if self.model_name.startswith("gpt"):
            try:
                response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0, 
                    )['choices'][0]['message']['content']
            except:
                response = ""
        else:
            completions = openai.Completion.create(
            engine = self.model_name,  # Determines the quality, speed, and cost.
            temperature = 0.5,            # Level of creativity in the response
            prompt=prompt,           # What the user typed in
            max_tokens=50,             # Maximum tokens in the prompt AND response
            n=1,                        # The number of completions to generate
            stop=None,                  # An optional setting to control response generation
        )
            response = completions.choices[0].text
        return response
        


        
    def __call__(self,text):
        response = self._call_api(text)
        ans = self.extract_ans(response)
        return ans
  
def get_model(model_name,src_lang,tgt_lang):
    if model_name in ValidHFModel:
        model = HFTranslator(model_name,src_lang,tgt_lang)
    elif model_name in ValidOpenAiModel.keys():
        model = OpenAiTranslator(model_name,src_lang,tgt_lang)
    else:
        raise ValueError("please enter a valid model name!")
    return model 


        


if __name__ == "__main__":
    text = "I don't speak German"
    translator = OpenAiTranslator("text-curie-001","English","German",few_shot=True)
    print(translator(text))
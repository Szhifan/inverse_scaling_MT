import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from utils import get_prefix,get_prefix_capitalize
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import re
import openai 
import torch 
from utils import id2lang,lang2id

device = "cpu"

# DEBUG: counting errors
error_count = 0
hf_cache_dir = "/Users/sunzhifan/.cache/huggingface/hub"
#Pre-define some valid Hugging Face models. 
ValidHFModel = [
    "mbart-large-50-many-to-one-mmt",
    "mbart-large-50-one-to-many-mmt",
    "flan-t5-small",
    "flan-t5-base",
    "flan-t5-large",
    "flan-t5-xl",
    "flan-t5-xxl",
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b"

]
mbart_lang_ids = {"English":"en_XX","German":"de_DE","Russian":"ru_RU","French":"fr_XX","Romanian":"ro_RO"}
class HFTranslator():
    
    def __init__(self,model_name,src_lang,tgt_lang,few_shot=False,use_prefix=False) -> None:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang 
        self.model_name = model_name
        self.need_prompt = True
        self.few_shot = few_shot
        self.use_prefix = use_prefix
        use_fast = True 
        if model_name.startswith("flan-"):
            prefix = "google/"
            torch.cuda.empty_cache()
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                prefix + model_name,
                cache_dir=hf_cache_dir,
                max_length=1024).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(
            prefix + model_name,
            cache_dir=hf_cache_dir
        )
        elif model_name.startswith("mbart"):
            self.need_prompt = False
            prefix = "facebook/"  
            self.model = MBartForConditionalGeneration.from_pretrained(prefix+model_name).to(device)
            self.tokenizer = MBart50TokenizerFast.from_pretrained(prefix+model_name)
            self.tokenizer.src_lang = mbart_lang_ids[self.src_lang]
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, max_length=1024,cache_dir=hf_cache_dir).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast,
            cache_dir=hf_cache_dir,
            model_max_length=1023)
      
        self.num_params = sum(p.numel() for p in self.model.parameters())
  
    def get_prompt(self,text):
        """
        Construct MT prompt format: 
        translate [src] to [tgt]: text 
        """
        src = self.src_lang
        tgt = self.tgt_lang 
        prompt = f"Translate {src} to {tgt}: {text}"
        return prompt 

    def __call__(self,src_text:str):
        if self.use_prefix:
            src_text = get_prefix(lang2id[self.src_lang])+src_text
        if self.need_prompt:
            prompt = self.get_prompt(src_text)
            input_ids = self.tokenizer(prompt,return_tensors="pt")["input_ids"].to(device)
            outputs = self.model.generate(input_ids,max_length=120)
        else:
            input_ids = self.tokenizer(src_text,return_tensors="pt")["input_ids"].to(device)
            if re.search(r"-to-many",self.model_name):
                outputs = self.model.generate(input_ids,forced_bos_token_id=self.tokenizer.lang_code_to_id[mbart_lang_ids[self.tgt_lang]])
            else:
                outputs = self.model.generate(input_ids)
        translation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translation[0]

ValidOpenAiModel ={
    "gpt-3.5-turbo":"175B",
    "text-davinci-001":"175B",
    "text-davinci-002":"175B",
    "text-davinci-003":"175B",
    "text-curie-001":"6.7B",
    "text-babbage-001":"1.3B",
    "text-ada-001":"350M",
    "davinci":"175B",
    "curie":"6.7B",
    "babbage":"1.3B",
    "ada":"350M"
}
class OpenAiTranslator():

    def __init__(self,model_name,src_lang,tgt_lang,few_shot=False,use_prefix=False) -> None:
        openai.api_key = ""

        self.model_name = model_name
        self.src_lang = src_lang 
        self.tgt_lang = tgt_lang
        self.num_params = ValidOpenAiModel[self.model_name]
        self.few_shot = few_shot # determine if few-shot-prmopt is needed. 
        self.use_prefix = use_prefix
    def get_prompt(self,src_text):
        """
        Construct MT prompt format: 
        translate [src] to [tgt]: text 
        """
        
        if self.few_shot: 
            prompt = f"Translate {self.src_lang} to {self.tgt_lang}:\n"
            prompt += self.construct_few_shot_examples() + f"\n[{self.src_lang}]:{src_text}\n[{self.tgt_lang}]:"
        else:
            prompt = f"Translate {self.src_lang} to {self.tgt_lang}: \"{src_text}\""
        return prompt
    def construct_few_shot_examples(self):
        example_pair = {"en":"I don't know","de":"Ich weiß nicht.","fr":"Je ne sais pas.","ro":"Nu știu.","ru":"Я не знаю.","zh":"我能不知道。"}
        src_example = example_pair[lang2id[self.src_lang]]
        tgt_example = example_pair[lang2id[self.tgt_lang]]
        context = f"[{self.src_lang}]:{src_example}\n[{self.tgt_lang}]:{tgt_example}"  
        return context

    def extract_ans(self,response):
        translation = re.sub(r"\n","",response)
        if self.few_shot: 
            try:
                if re.search(r"^(.+?)\[.+?\]",translation):
                    translation = re.search(r"^(.+?)\[.+?\]",translation).group(1)
            except: 
                pass 
        translation = re.sub(r"\n","",translation)
        return translation


    def _call_api(self,text):
        if self.use_prefix:
            text = get_prefix(lang2id[self.src_lang])+text
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

            max_tokens=120,             # Maximum tokens in the prompt AND response

            n=1,                        # The number of completions to generate
            stop=None,                  # An optional setting to control response generation
        )
            response = completions.choices[0].text
        return response      
    def __call__(self,text):
        for _ in range(4):
            try:
                response = self._call_api(text)

                ans = self.extract_ans(response)
                return ans 
            except:
                pass 
        return None
ValidLlamaModel = [] #a list (or dict) of all valid llama models.
class LlamaTranslator():
    """
    A class of llama translator that can translate given text by calling the instance itself. 
    """
    def __init__(self,model_name,src_lang,tgt_lang,few_shot,use_prefix) -> None:
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.few_shot = few_shot
        self.use_prefix = use_prefix 
    
    def get_prompt(self,src_text):
        """
        Construct MT prompt format (other formats could be explore according to the performance
        of llama model): 
        translate [src] to [tgt]: text 
        """
        
        if self.few_shot: 
            prompt = f"Translate {self.src_lang} to {self.tgt_lang}:\n"
            prompt += self.construct_few_shot_examples() + f"\n[{self.src_lang}]:{src_text}\n[{self.tgt_lang}]:"
        else:
            prompt = f"Translate {self.src_lang} to {self.tgt_lang}: \"{src_text}\""
        return prompt
    def construct_few_shot_examples(self):
        """
        This function constructs one-shot prompts in the format: 
        [src_lang]:[src_text]
        [tgt_lang]:[tgt_text]
        """
        example_pair = {"en":"I don't know","de":"Ich weiß nicht.","fr":"Je ne sais pas.","ro":"Nu știu.","ru":"Я не знаю.","zh":"我能不知道。"}
        src_example = example_pair[lang2id[self.src_lang]]
        tgt_example = example_pair[lang2id[self.tgt_lang]]
        context = f"[{self.src_lang}]:{src_example}\n[{self.tgt_lang}]:{tgt_example}"  
        return context
    def extract_ans(self,response):
        """
        clean the response from the model if it outputs irrelevant stuffs. 
        """
        ... 
    def __call__(self,text):
        """
        this functions does the actual translation. You could finish the rest of the function 
        according to how llama models are called. 
        """
        prompt = self.get_prompt(text)
        ...
        ans = self.extract_ans(response)
        return ans 
         
def get_model(model_name,src_lang,tgt_lang,few_shot,use_prefix):
    if model_name in ValidHFModel:
        model = HFTranslator(model_name,src_lang,tgt_lang,few_shot=few_shot,use_prefix=use_prefix)
    elif model_name in ValidOpenAiModel.keys():
        model = OpenAiTranslator(model_name,src_lang,tgt_lang,few_shot=few_shot,use_prefix=use_prefix)
    else:
        raise ValueError("please enter a valid model name!")
    return model
if __name__ == "__main__":

    text_en = "I am your vather"
    text_de = "was is die Hauptstadt von Russland?"
    text_zh = "日本的首都在哪里？"
    translator = OpenAiTranslator("text-babbage-001","English","German",use_prefix=False)
    print(translator(text_en))

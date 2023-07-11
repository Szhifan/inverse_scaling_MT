import re
lang_pair = "de_en"
model = "nmslgpt"
dataset = "datasets/truthfullqa/en_de.df"
translation_output_dir = re.sub(r"[a-z]{2}_[a-z]{2}\.df",f"{lang_pair}_output",dataset)+f"/{model}.txt"
print(translation_output_dir)
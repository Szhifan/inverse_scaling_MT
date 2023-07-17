from utils import eval
import re 
import os
dir_de = "datasets/truthfullqa/en_de_output"
ref_de = "datasets/truthfullqa/ref_de.txt"
for f in os.listdir(dir_de):
    if re.search(r"001",f):
        print(f)
        info = eval(ref_de,dir_de+"/"+f)
        print(info)

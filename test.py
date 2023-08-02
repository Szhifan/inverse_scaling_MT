from utils import * 
ref_path = "truthfullqa/ref_en.txt"
mt_path = "truthfullqa/prefix/ru_en_output/text-davinci-003.txt"
print(eval(ref_path,mt_path))
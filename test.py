from utils import * 
ref_dir = "truthfullqa/ref_de.txt"
mt_dir = "truthfullqa/prompt2/en_de_output/ada.txt"
print(eval(ref_dir,mt_dir)) 
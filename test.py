dir = "datasets/truthfullqa/en_de_output/text-babbage-001.txt"
list = [s for s in open(dir,"r").readlines() if s!="\n"]
print(list)


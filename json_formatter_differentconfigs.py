import json
import os

f = open('C:/Users/matte/OneDrive/Desktop/Thesis/Thesis_Code/datasets/MalteseFood_Dataset_Final/resized/combined/annotation.json')
root_dir = r'C:\Users\matte\OneDrive\Desktop\Thesis\Thesis_Code\datasets\MalteseFood_Dataset_Final\resized\combined'
config_dir = ['/Config1/train', '/Config1/test', '/Config2/train', '/Config2/test', '/Config3/train', '/Config3/test']
jsonFile = {}
data_set = json.load(f)

for cls in config_dir:
    allFileNames = os.listdir(root_dir+cls)
    for name in allFileNames:
        if not (data_set.get(name) is None):
            jsonFile[name] = data_set.get(name)
    with open(
            root_dir+cls+'/annotation.json',
            'w', encoding='utf-8') as f:
        json.dump(jsonFile, f, ensure_ascii=False, indent=4)
    jsonFile = {}

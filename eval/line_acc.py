import pandas as pd
import re
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--type', type=str)
args = parser.parse_args()
model = args.model
modeltype = args.type
if modeltype == "base":
    modeltype = ""

dirpath = f"../{modeltype}Results/BigVul/line/{model}/"
#获取文件列表
# folder_list = os.listdir(dirpath)
folder_list = ['basic', 'basic2', "cot", 'cot2']
# folder_list = ['basic']

for folder in folder_list:
    df1 = pd.read_csv(dirpath + folder + "/lineformatted.csv", header=0, encoding='utf-8')
    res_list = df1['response'].tolist()
    # 正则匹配所有的YES不区分大小写
    Tnum = len(re.findall(r'yes', ' '.join(res_list), re.IGNORECASE))
    Fnum = len(res_list) - Tnum

    ACC = round(Tnum / 100, 2)
    
    print(f"{model} {folder} ACC:{ACC}")

    
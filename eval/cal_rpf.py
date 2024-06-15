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


dirpath = f"../{modeltype}Results/BigVul/tc/{model}/"
#获取文件列表
# folder_list = os.listdir(dirpath)
# folder_list = ['basic', 'basic2', "cot", 'cot2']
folder_list = ['basic']
# folder_list = ['basic2', "cot", 'cot2']
# folder_list = ['basic2', "cot", 'cot2']

for folder in folder_list:
    df1 = pd.read_csv(dirpath + folder + "/vul1formatted.csv", header=0, encoding='utf-8')
    res_list = df1['response'].tolist()
    # 正则匹配所有的YES不区分大小写
    TP = len(re.findall(r'yes', ' '.join(res_list), re.IGNORECASE))
    FN = len(res_list) - TP
    df0 = pd.read_csv(dirpath + folder + "/vul0formatted.csv", header=0, encoding='utf-8')
    res_list = df0['response'].tolist()
    # 正则匹配所有的YES不区分大小写
    FP = len(re.findall(r'yes', ' '.join(res_list), re.IGNORECASE))
    TN = len(res_list) - FP
    P = round(TP / (TP + FP), 3)
    R = round(TP / (TP + FN), 3)
    F1 = round(2 * P * R / (P + R), 3)
    print(f"{model} {folder} P:{P}, R:{R}, F1:{F1}")
    # 输出TP, FP, TN, FN
    print(f"{model} {folder} TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}")
    
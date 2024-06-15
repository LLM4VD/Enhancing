import transformers
import torch
import torch.nn as nn
import pandas as pd
import os
import json
import openai
import argparse
from create_message import get_message
from transformers import AutoTokenizer, AutoModelForCausalLM


parser = argparse.ArgumentParser(description="Process some codes with codegm model.")
parser.add_argument('--tp', type=str, help='The template flag to use: basic, basic1, basic2, cot, cot1, cot2')
parser.add_argument('--type', type=str,help='The template flag to use: basic, basic1, basic2, cot, cot1, cot2')
args = parser.parse_args()
templateflag = args.tp

modeltype = args.type

if modeltype == "lora" or modeltype == "qlora":
    model_path = f"../codegm/{modeltype}/ftmodel"
if modeltype == "base":
    model_path = "../codegm/base"
    modeltype = ""


client = openai.OpenAI(
    api_key="xxx",
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# model_id = "/data/wyk/Huawei/LLMs/codegm/base"

dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    torch_dtype=dtype,
)
def get_response(code, templateflag):
    chat = get_message(code, templateflag)
    # chat = chat[1:]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=512, do_sample=False)
    prompt_len = inputs.shape[-1]  # Access the length directly from the tensor
    response = tokenizer.decode(outputs[0][prompt_len:],skip_special_tokens=True)

    return response


def format_answer(answer):
    
    completion = client.chat.completions.create(
                    # model="gpt-3.5-turbo-1106",
                    model="gpt-4o",
                    response_format={"type": "json_object"},
                    messages=[
      {'role': 'user', 'content': "The following is an answer to determine whether a piece of code is a vulnerability. "
                   "You need to summarize the answer and determine whether what it expresses has a vulnerability. "
                   "Your response should only be Yes or No."
                   "Response like json format:\n"
                   "{\"VULNERABLE\": \"YES/NO\"}\n"
                   "If you cannot judge the content, treat it as No.\n"
                   "###ANSWER\n"
                   + answer + "\n"
 }
    ]
     )
    return completion.choices[0].message.content



def process_code_list(code_list, templateflag):
    results = []
    i= 0
    for code in code_list:
        response = get_response(code, templateflag)
        results.append({"code": code, "response": response})
        i += 1
        print(i)
    return results

def main():


    # 读取CSV文件中的代码列表
    df_vul1 = pd.read_csv("testdata/BigVul/512tokenvul1.csv", header=0, encoding='utf-8')
    df_vul1 = df_vul1.iloc[:100]  # 只处理前10行
    code_list = df_vul1['func_before'].tolist()

    responses = process_code_list(code_list, templateflag)

        
    outputdir = f"./{modeltype}Results/BigVul/tc/codegm/"+ templateflag
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    # 保存响应到 JSON 文
    with open(outputdir+"/codegmvul1.json", 'w') as f:
        json.dump(responses, f, indent=2)

    print(f"Responses saved to {outputdir}+/codegmvul1.json")
    formatted_ans = []
    for ans in responses:
        formatted_ans.append(format_answer(ans['response']))
    # 保存到 CSV 文件
    df = pd.DataFrame(formatted_ans, columns=['response'])
    df.to_csv(f"{outputdir}/vul1formatted.csv", index=False)
    
    
        # 读取CSV文件中的代码列表
    df_vul0 = pd.read_csv("testdata/BigVul/512tokenvul0.csv", header=0, encoding='utf-8')
    df_vul0 = df_vul0.iloc[:100]  # 只处理前10行
    code_list = df_vul0['func_before'].tolist()

    responses = process_code_list(code_list, templateflag)
    
    # 保存响应到 JSON 文件
    with open(outputdir+"/codegmvul0.json", 'w') as f:
        json.dump(responses, f, indent=2)

    print(f"Responses saved to {outputdir}/codegmvul0.json")
    formatted_ans = []
    for ans in responses:
        formatted_ans.append(format_answer(ans['response']))
    # 保存到 CSV 文件
    df = pd.DataFrame(formatted_ans, columns=['response'])
    df.to_csv(f"{outputdir}/vul0formatted.csv", index=False)
    
    
main()
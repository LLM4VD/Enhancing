import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import pandas as pd
import os
import json
import openai
import argparse
from create_message import get_message



parser = argparse.ArgumentParser(description="Process some codes with llama model.")
parser.add_argument('--tp', type=str, help='The template flag to use: basic, basic1, basic2, cot, cot1, cot2')
parser.add_argument('--type', type=str,help='The template flag to use: basic, basic1, basic2, cot, cot1, cot2')
args = parser.parse_args()
templateflag = args.tp

modeltype = args.type

if modeltype == "lora" or modeltype == "qlora":
    model_path = f"../llama/{modeltype}/ftmodel"
if modeltype == "base":
    model_path = "../llama/base"
    modeltype = ""


client = openai.OpenAI(
    api_key="xxx",
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

def get_response(code, templateflag):
    conversation = get_message(code, templateflag)
    prompt = pipeline.tokenizer.apply_chat_template(
    conversation, 
    tokenize=False, 
    add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False
    )
    
    response = outputs[0]["generated_text"][len(prompt):]
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

        
    outputdir = f"./{modeltype}Results/BigVul/tc/llama/"+ templateflag
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    # 保存响应到 JSON 文
    with open(outputdir+"/llamavul1.json", 'w') as f:
        json.dump(responses, f, indent=2)

    print(f"Responses saved to {outputdir}/llamavul1.json")
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
    with open(outputdir+"/llamavul0.json", 'w') as f:
        json.dump(responses, f, indent=2)

    print(f"Responses saved to {outputdir}/llamavul0.json")
    formatted_ans = []
    for ans in responses:
        formatted_ans.append(format_answer(ans['response']))
    # 保存到 CSV 文件
    df = pd.DataFrame(formatted_ans, columns=['response'])
    df.to_csv(f"{outputdir}/vul0formatted.csv", index=False)
    
    
main()
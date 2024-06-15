import transformers
import torch
import torch.nn as nn
import pandas as pd
import os
import json
import openai
import argparse
from create_line_message import get_line_message
from transformers import AutoTokenizer, AutoModelForCausalLM


parser = argparse.ArgumentParser(description="Process some codes with codegm model.")
parser.add_argument('--tp', type=str, help='The template flag to use: basic, basic1, basic2, cot, cot1, cot2')
parser.add_argument('--type', type=str,help='The template flag to use: basic, basic1, basic2, cot, cot1, cot2')
args = parser.parse_args()
templateflag = args.tp

modeltype = args.type

if modeltype == "lora" or modeltype == "qlora":
    model_id = f"../codegm/line/{modeltype}/ftmodel"
if modeltype == "base":
    model_id = "../codegm/base"
    modeltype = ""



client = openai.OpenAI(
    api_key="xxxx",
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# model_id = "/data/wyk/Huawei/LLMs/codegm/base"

dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype,
)
def get_response(code, templateflag):
    chat = get_line_message(code, templateflag)
    # chat = chat[1:]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=512, do_sample=False)
    prompt_len = inputs.shape[-1]  # Access the length directly from the tensor
    response = tokenizer.decode(outputs[0][prompt_len:],skip_special_tokens=True)

    return response


def format_answer(code, answer,lines):
    
    completion = client.chat.completions.create(
                    # model="gpt-3.5-turbo-1106",
                    model="gpt-4o",
                    # response_format={"type": "json_object"},
                    messages = [
    {
        "role": "system",
        "content": "You are an AI assistant specialized in code vulnerability detection. Your task is to determine whether the answer regarding the location of the vulnerability lines of code is correct or not."
    },
    {
        "role": "user",
        "content": "Please determine whether the inferred ANSWER reasonably identifies the TRULY VULNERABILITY LINES. The ANSWER is correct if it contains at least one line of code that that matches the TRULY VULNERABILITY LINES. Respond only with 'Yes' or 'No':\n\n" +
                   "VULNERABILITY CODE:\n" +
                   code + "\n\n" +
                   "TRULY VULNERABILITY LINES:\n" +
                   lines + "\n\n" +
                   "ANSWER:\n" +
                   answer + "\n\n" +
                   "Respond with one word only: YES or NO."
    }
]

     )
    return completion.choices[0].message.content



def process_code_list(code_list, line_list, templateflag):
    results = []
    for i in range(len(code_list)):
        response = get_response(code_list[i], templateflag)
        results.append({"code": code_list[i], "response": response, "truly_vulnerable_lines":line_list[i]})
        print(i)
    return results

def main():
    outputdir = f"./{modeltype}Results/BigVul/line/codegm/"+ templateflag
    # 读取CSV文件中的代码列表
    df_vul1 = pd.read_csv("testdata/BigVul/linevul1.csv", header=0, encoding='utf-8')
    df_vul1 = df_vul1.iloc[:100]  # 只处理前10行
    code_list = df_vul1['func_before'].tolist()
    line_list = df_vul1['lines_before'].tolist()

    responses = process_code_list(code_list, line_list, templateflag)

        
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    # 保存响应到 JSON 文
    with open(outputdir+"/codegmline.json", 'w') as f:
        json.dump(responses, f, indent=2)

    print(f"Responses saved to {outputdir}/codegmline.json")
    formatted_ans = []
    for ans in responses:
        formatted_ans.append(format_answer(ans['code'],ans['response'],ans['truly_vulnerable_lines']))
    # 保存到 CSV 文件
    df = pd.DataFrame(formatted_ans, columns=['response'])
    df.to_csv(f"{outputdir}/lineformatted.csv", index=False)
    
    
main()
import transformers
import torch
import torch.nn as nn
import pandas as pd
import os
import json
import openai
import argparse
from create_line_message import get_line_message

parser = argparse.ArgumentParser(description="Process some codes with llama model.")
parser.add_argument('--tp', type=str, help='The template flag to use: basic, basic1, basic2, cot, cot1, cot2')
parser.add_argument('--type', type=str,help='The template flag to use: basic, basic1, basic2, cot, cot1, cot2')
args = parser.parse_args()
templateflag = args.tp

modeltype = args.type

if modeltype == "lora" or modeltype == "qlora":
    model_id = f"../llama/line/{modeltype}/ftmodel"
if modeltype == "base":
    model_id = "../llama/base"
    modeltype = ""



client = openai.OpenAI(
    api_key="xxx",
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# model_id = "/data/wyk/Huawei/LLMs/llama/base"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

def get_response(code, templateflag):
    conversation = get_line_message(code, templateflag)
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
    outputdir = f"./{modeltype}Results/BigVul/line/llama/"+ templateflag

    # 读取CSV文件中的代码列表
    df_vul1 = pd.read_csv("testdata/BigVul/linevul1.csv", header=0, encoding='utf-8')
    df_vul1 = df_vul1.iloc[:100]  # 只处理前10行
    code_list = df_vul1['func_before'].tolist()
    line_list = df_vul1['lines_before'].tolist()

    responses = process_code_list(code_list, line_list, templateflag)


    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    # 保存响应到 JSON 文
    with open(outputdir+"/llamaline.json", 'w') as f:
        json.dump(responses, f, indent=2)

    print(f"Responses saved to {outputdir}/llamaline.json")
    formatted_ans = []
    for ans in responses:
        formatted_ans.append(format_answer(ans['code'],ans['response'],ans['truly_vulnerable_lines']))
    # 保存到 CSV 文件
    df = pd.DataFrame(formatted_ans, columns=['response'])
    df.to_csv(f"{outputdir}/lineformatted.csv", index=False)
    
    
main()
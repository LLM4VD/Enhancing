import openai
from openai import OpenAI
import pandas as pd
import json
import time
import os
import argparse
from create_line_message import get_line_message



client = openai.OpenAI(
    api_key="xxx",
)

def gpt_infer(code, templateflag):
    conversation = get_line_message(code, templateflag)
    while True:
        try:
            completion = client.chat.completions.create(
                # model="gpt-4-1106-preview",
                model="gpt-4o",
                # response_format={"type": "json_object"},
                messages=conversation
            )
            return completion.choices[0].message.content
        except openai.APITimeoutError:
            print("Request timed out. Retrying...")
            time.sleep(10)  # 等待5秒后重试
        except openai.OpenAIError as e:
            print(f"An error occurred: {e}")
            break
    return None


def format_answer(code, answer,lines):
    
    completion = client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    # model="gpt-4o",
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
        response = gpt_infer(code_list[i], templateflag)
        results.append({"code": code_list[i], "response": response, "truly_vulnerable_lines":line_list[i]})
        print(i)
    return results

def main():
    parser = argparse.ArgumentParser(description="Process some codes with GPT model.")
    parser.add_argument('--tp', type=str, help='The template flag to use: basic, basic1, basic2, cot, cot1, cot2')
    args = parser.parse_args()

    templateflag = args.tp

    # 读取CSV文件中的代码列表
    df_vul1 = pd.read_csv("testdata/BigVul/linevul1.csv", header=0, encoding='utf-8')
    df_vul1 = df_vul1.iloc[:100]  # 只处理前10行
    code_list = df_vul1['func_before'].tolist()
    line_list = df_vul1['lines_before'].tolist()

    responses = process_code_list(code_list, line_list, templateflag)

        
    outputdir = "./Results/BigVul/line/gpt/"+ templateflag
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    # 保存响应到 JSON 文
    with open(outputdir+"/gptline.json", 'w') as f:
        json.dump(responses, f, indent=2)

    print(f"Responses saved to {outputdir}/gptline.json")
    formatted_ans = []
    for ans in responses:
        formatted_ans.append(format_answer(ans['code'],ans['response'],ans['truly_vulnerable_lines']))
    # 保存到 CSV 文件
    df = pd.DataFrame(formatted_ans, columns=['response'])
    df.to_csv(f"{outputdir}/lineformatted.csv", index=False)

if __name__ == "__main__":
    main()


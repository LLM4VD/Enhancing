import pandas as pd
import os
import json
import openai
import argparse

client = openai.OpenAI(
    api_key="xxx",
)


parser = argparse.ArgumentParser()
parser.add_argument('--tp', type=str, help='The template flag to use: basic, basic1, basic2, cot, cot1, cot2')
parser.add_argument('--model', type=str, help='The template flag to use: basic, basic1, basic2, cot, cot1, cot2')
parser.add_argument('--type', type=str,help='The template flag to use: basic, basic1, basic2, cot, cot1, cot2')
args = parser.parse_args()
model = args.model
template = args.tp
if args.type == "base":
    modeltype = ""
else:
    modeltype = args.type

filepath = f"../{modeltype}Results/BigVul/line/{model}/{template}"

def format_answer(code, answer,lines):
    
    completion = client.chat.completions.create(
                    # model="gpt-3.5-turbo-1106",
                    # model="gpt-3.5-turbo-0125",
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


with open(f'{filepath}/{model}line.json', 'r') as file:
    # 读取JSON数据
    data = json.load(file)

# df = pd.read_csv("../testdata/Bigvul/linevul1.csv",header=0)

formatted_ans = []

for d in data:
    formatted_ans.append(format_answer(d['code'],d['response'],d['truly_vulnerable_lines']))
    
df = pd.DataFrame(formatted_ans, columns=['response'])
df.to_csv(f"{filepath}/lineformatted.csv", index=False)
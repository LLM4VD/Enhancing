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

filepath = f"../{modeltype}Results/BigVul/tc/{model}/{template}"

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

format_file = f'{filepath}/{model}vul1.json'
with open(format_file, 'r') as file:
    # 读取JSON数据
    data = json.load(file)

# df = pd.read_csv("../testdata/Bigvul/linevul1.csv",header=0)
print(f"format file:{format_file}")

formatted_ans = []

for d in data:
    formatted_ans.append(format_answer(d['response']))
    
df = pd.DataFrame(formatted_ans, columns=['response'])
df.to_csv(f"{filepath}/vul1formatted.csv", index=False)

format_file = f'{filepath}/{model}vul0.json'
with open(format_file, 'r') as file:
    # 读取JSON数据
    data = json.load(file)

# df = pd.read_csv("../testdata/Bigvul/linevul1.csv",header=0)

formatted_ans = []
print(f"format file:{format_file}")
for d in data:
    formatted_ans.append(format_answer(d['response']))
    
df = pd.DataFrame(formatted_ans, columns=['response'])
df.to_csv(f"{filepath}/vul0formatted.csv", index=False)
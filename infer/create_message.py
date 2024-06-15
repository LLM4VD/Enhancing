
def get_message(code,templateflag):
    system_message = {'role': 'system', 'content': 'You are an AI assistant specialized in code vulnerability prediction.\
    Now you need to identify whether a function contains a vulnerability or not.'}
    with open("./templates/basic.txt","r") as f:
        basic_prompt = f.read()
    with open("./templates/cotprompt.txt","r") as f:
        cot_prompt = f.read()   
    with open("./templates/code1.txt","r") as f:
        code1 = f.read()
    with open("./templates/code2.txt","r") as f:
        code2 = f.read()
    with open("./templates/cotanswer1.txt","r") as f:
        cotanswer1 = f.read()
    with open("./templates/cotanswer2.txt","r") as f:
        cotanswer2 = f.read()
    
    # basic
    if templateflag == 'basic':
        messages = [system_message, {'role': 'user', 'content': basic_prompt+"\n###Code\n"+code}]
    # basic1shot
    elif templateflag == 'basic1':
        shot1 = basic_prompt + "\n" + code1
        query_promot = basic_prompt + "\n" + code
        messages = [system_message, 
                    {'role': 'user', 'content': shot1},
                    {'role': 'assistant', 'content': """###Answer1: 
                     The function has a buffer overflow vulnerability due to the code: 
    if (l_strnstart("MSG", 4, (const char *)bp, length)) /* A REQuest */ ND_PRINT((ndo, " BEEP MSG")); If the length of bp is less than 4, the function will read beyond the buffer boundary, \
        leading to a buffer overflow vulnerability.
    {"VULNERABLE": "YES"}
    """},
                    {'role': 'user', 'content':query_promot}
                    ]
    elif templateflag == "basic2":
        shot1 = basic_prompt + "\n" + code1
        shot2 = basic_prompt + "\n" + code2
        query_prompt = basic_prompt + "\n###Code\n" + code
        messages = [system_message, 
                    {'role': 'user', 'content': shot1},
                    {'role': 'assistant', 'content': """###Answer1: 
                     The function has a buffer overflow vulnerability due to the code: 
    if (l_strnstart("MSG", 4, (const char *)bp, length)) /* A REQuest */ ND_PRINT((ndo, " BEEP MSG")); If the length of bp is less than 4, the function will read beyond the buffer boundary, \
        leading to a buffer overflow vulnerability.
    {"VULNERABLE": "YES"}
    """},
                    {'role': 'user', 'content': shot2},
                    {'role': 'assistant', 'content': """###Answer2:
                     The memset function is used to initially set the entire buffer to zero, ensuring there's no lingering junk data. \
        The function doesn't take any untrusted inputs from the outside. The only input is mac_addr, and we assume the code calling this function ensures it's a valid MAC address.\
            There is no apparently vulnerability in the function.
    {"VULNERABLE": "NO"}"""},
                    {'role': 'user', 'content':query_prompt}
                    ]
        
    elif templateflag == "cot":
        messages = [system_message, {'role': 'user', 'content': cot_prompt+"\n###Code\n"+code}]
    
    elif templateflag == "cot1":
        shot1 = cot_prompt + "\n" + code1
        messages = [system_message, {'role': 'user', 'content': shot1},
                    {'role': 'assistant', 'content': cotanswer1},
                    {'role': 'user', 'content': cot_prompt+"\n###Code\n"+code}
                    ] 
    
    elif templateflag == "cot2":
        shot1 = cot_prompt + "\n" + code1
        shot2 = cot_prompt + "\n" + code2
        messages = [system_message, {'role': 'user', 'content': shot1},
                    {'role': 'assistant', 'content': cotanswer1},
                    {'role': 'user', 'content': shot2},
                    {'role': 'assistant', 'content': cotanswer2},
                    {'role': 'user', 'content': cot_prompt+"\n###Code\n"+code}
                    ] 
    return messages
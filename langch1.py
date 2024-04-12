from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import datetime
from rich.console import Console
from rich.markdown import Markdown
from rich import print as rprint
import subprocess
import os
from pathlib import Path
from textwrap import dedent

console = Console()
#llm2 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, base_url='http://localhost:1234/v1')


def make_fullpath(fn: str) -> Path:
    return Path.home() / 'Documents' / fn


def save_and_execute_python(s: str):
    console.print('executing code...', style='red')
    full_path = make_fullpath('temp.py')
    with open(full_path, 'w') as f:
        f.write(s)
    
    result = subprocess.run(["python", full_path], capture_output=True)
    output = result.stdout.decode("utf-8")
    err = result.stderr.decode("utf-8")
    if len(output) > 0:
        console.print(Markdown(output), style='yellow')
    else:
        console.print(err, style='red')
    return output, err


def save_and_execute_bash(s: str):
    console.print('executing code...', style='red')
    full_path = make_fullpath('temp')
    with open(full_path, 'w') as f:
        f.write(s)
    
    os.system(f'chmod +x {full_path}')
    result = subprocess.run([full_path], shell=True, executable='/bin/bash', capture_output=True)
    output = result.stdout.decode("utf-8")
    err = result.stderr.decode("utf-8")
    if len(output) > 0:
        console.print(Markdown(output), style='yellow')
    else:
        console.print(err, style='red')
    return output, err


def extract_code_block(contents: str, sep: str) -> str:
    '''extracts the string between a pair of separtors. If there is no seperator return None'''
    y = len(sep)
    start_index = contents.find(sep)
    if start_index >= 0:
        end_index = contents.find(sep, start_index + y)
        if end_index >= 0:
            return contents[start_index + y:end_index]
    return None


def execute_script(code):
    output = None
    err = None
    msg = None
    if code.startswith('python\n'):
        output, err = save_and_execute_python(code[7:])
        if err:
            msg = err
        else:
            msg = output
    elif code.startswith('bash\n'):
        output, err = save_and_execute_bash(code[5:])
        msg = output
    else:
        console.print('code block found but not executed\n' + code, style='yellow')
    return msg


def strip_content_after_block(s: str, sep: str) -> str:
    '''strip content after code block because the LLM often guesses the output of the script which confuses the LLM when it sees it in the chat history'''
    x = s.find(sep)
    y = s.find(sep, x+1)
    console.print('ignoring content\n' + s[y+3:], style='yellow')
    return s[:y]


def execute_code_reply(messages, retries: int):
    '''check if last message has a code block and automatically execute it and pass the results back to the llm'''
    msg = messages[-1]
    code = extract_code_block(msg.content, '```')
    executions = 0
    while code and executions < retries:
        msg.content = strip_content_after_block(msg.content, '```')
        executions += 1
        output = execute_script(code)
        if output:
            messages.append(HumanMessage(f'script output:\n```\n{output}\n```\n'))
            msg = llm.invoke(messages)
            messages.append(msg)
            code = extract_code_block(msg.content, '```')
        else:
            code = None


def print_history(messages):
    for m in messages:
        c = 'cyan'
        role = 'assistant'
        if isinstance(m, SystemMessage):
            c = 'red'
            role = 'system'
        elif isinstance(m, HumanMessage):
            c = 'green'
            role = 'user'
            
        md = Markdown(m.content)
        console.print(f'\n{role}:', style=c)
        console.print(md, style=c)

messages = [
    SystemMessage(f'You are Marvin a super intelligent AI chatbot trained by OpenAI. You are hepful and concise. The current datetime is {datetime.datetime.now().isoformat()}. Always write code within markdown code blocks'),
    HumanMessage('use bash to show .csv files in ~/Downloads sorted by filename')
    # HumanMessage('write a bash script to get the details of the Linux distro and kernel version')
#     HumanMessage(dedent("""
# You are an intelligent system capable of solving optimization problems. I have a problem that requires you to select a combination of coins that exactly pay for an apple that costs 6 pence (p). The available coins are: one 10p, one 5p, and three 2p. Your task is to determine the combination of coins that will exactly pay for the 6p apple. Please provide the specific coins that should be used, along with the total value of the coins selected. Solve this problem and provide a detailed step-by-step explanation of your approach and the final solution."
#                         """))
    # HumanMessage(dedent("""
    # I need pay for an apple that costs 6p. I have the following coins: one 10p, one 5p, three 2p. Select the coins that exactly pay for the apple. You can only select coins from those available.
    # Write your answer as
    # ## thinking
    #     - review the problem and how to approach it
    # ## solution
    #     - write your solution here                
    #                     """))
#    HumanMessage('what is 58461307 / 7643. write the request')
    # HumanMessage('how many years ago was the first moon landing')
]


msg = llm.invoke(messages)
messages.append(msg)

execute_code_reply(messages, 3)

print_history(messages)

# messages.append(HumanMessage('<tool_response>7649<tool_response>'))

# msg = llm.invoke(messages)
# messages.append(msg)

# print_history(messages)

# messages.append(HumanMessage('your solution is not correct. please fix.'))

# msg = llm.invoke(messages)
# messages.append(msg)

# print_history(messages)

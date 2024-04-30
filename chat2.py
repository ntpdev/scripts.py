#!/usr/bin/env python3
import argparse
from dataclasses import dataclass, asdict
from typing import List
from dataclasses_json import dataclass_json
import datetime
#from datetime import datetime, date, time
from openai import OpenAI
import os
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich import print as rprint
import requests
import json
import subprocess
from math import sqrt

# pip install dataclasses-json

model_name = {'gpt35':'gpt-3.5-turbo', 'gpt4':'gpt-4-turbo', 'llama3':'meta-llama/Llama-3-70b-chat-hf', 'ollama':'llama3'}
FNAME = 'chat-log.json'
console = Console()
FNCALL_SYSMSG = """
You are Marvin, an AI chatbot trained by OpenAI. You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: 
<tools>{"type": "function", "function": {"name": "eval", "description": "evaluates a mathematical expression and returns the result example 5 * 4 + 3 .\n\n    Args:\n    code (str): a mathematical expression.\n\n    Returns:\n    str: a number.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}</tools>
You call a tool by outputting json within <tool_call> tags. The result will be provided to you within <tool_return> tags.
"""

tool_ex = """
<tool_call>
{'arguments': {'code': '(datetime.now() - datetime(1969, 7, 20)).days // 365'}, 'name': 'eval'}
</tool_call>
"""

TOOL_FN = [ {
  "type": "function",
  "function": {
    "name": "evaluate",
    "description": "Evaluate a one line mathematical expression and return the result as a string",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "The mathematical expression to be evaluated. The expression can contain numbers, operators (+, -, *, /), parentheses and functions from python math module."
        }
      },
      "required": ["expression"]
    } } }]


q = """```python
from math import sqrt
x = 3
y = 4
print(sqrt(x ** 2 + y ** 2))
```
"""

@dataclass_json
@dataclass(frozen=True)
# add validation to check that role and content are not empty   
class ChatMessage:
    role: str
    content: str

    def __post_init__(self):
        if not self.role:
            raise ValueError("Role cannot be empty")
        if not self.content:
            raise ValueError("Content cannot be empty")


@dataclass
class CodeBlock:
    language: str
    lines: list[str]


def make_fullpath(fn: str) -> Path:
    return Path.home() / 'Documents' / fn


def is_toolcall(s: str) -> str:
    start = s.find('<tool_call>')
    if start >= 0:
        end = s.find('</tool_call>')
        contents = s[start+11:end].strip()
        contents = contents.replace('\'', '\"')
        print(contents)
#        data = json.loads(contents)
#        print(data['code'])
        x = '<tool_response>' + eval('datetime.now()').__repr__() + '</tool_response>'
        print(x)
        return x
    return None


def save_and_execute_python(code: list[str]):
    console.print('executing code...', style='red')
    full_path = make_fullpath('temp.py')
    with open(full_path, 'w') as f:
        f.write('\n'.join(code.lines))
    
    result = subprocess.run(["python", full_path], capture_output=True)
    output = result.stdout.decode("utf-8")
    err = result.stderr.decode("utf-8")
    if len(output) > 0:
        console.print(Markdown(output), style='yellow')
    else:
        console.print(err, style='red')
    return output, err


def save_and_execute_bash(code: list[str]):
    console.print('executing code...', style='red')
    full_path = make_fullpath('temp')
    with open(full_path, 'w') as f:
        f.write('\n'.join(code.lines))
    
    os.chmod(full_path, 0o755)  # Set executable permissions
    result = subprocess.run(["bash", full_path], capture_output=True)
    output = result.stdout.decode("utf-8")
    err = result.stderr.decode("utf-8")
    if len(output) > 0:
        console.print(Markdown(output), style='yellow')
    else:
        console.print(err, style='red')
    return output, err


def prt(msg : ChatMessage):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "cyan",
        "function": "yellow" }
    console.print(f'{msg.role}:\n', style=role_to_color[msg.role])
    md = Markdown(msg.content)
    console.print(md, style=role_to_color[msg.role])


def save(xs, filename):
    with open(filename, 'w') as f:
        f.write(ChatMessage.schema().dumps(xs, many=True))


def load_msg(s: str) -> str:
    xs = s.split()
    with open(make_fullpath(xs[1]), 'r') as f:
        return f.read()
    

def tool_response(s: str) -> str:
    xs = s[6:].strip()
    return '<tool_response>\n{"name": "eval", "content": "xxx"}\n</tool_response>\n'.replace('xxx', xs)


def load(filename: str) -> List[ChatMessage]:
    with open(filename, 'r') as f:
        return ChatMessage.schema().parse_raw(f.read())


def process_tool_call(tool_call):
    fnname = tool_call.function.name 
    args = json.loads(tool_call.function.arguments)
    console.print(f'tool call {fnname} {args}', style='yellow')
    if fnname == 'evaluate':
        r = eval(args['expression'])
        console.print(f'result = {str(r)}', style='yellow')
        d = {"role": "function",
             "name": fnname,
             "tool_call_id": tool_call.id,   
             "content": str(r)}
        return d
    return None


def extract_code_block(contents: str, sep: str) -> str:
    '''extracts the first code block'''
    xs = contents.splitlines()
    inside = False
    code = None
    for x in xs:
        if x.startswith(sep):
            inside = not inside
            if inside:
               code = CodeBlock(x[len(sep):], [])
            else:
                # if haven't found a language look in the contents
                s = contents.lower()
                if len(code.language) == 0:
                    if 'python' in s or 'print(' in s:
                        code.language = 'python'
                    elif 'bash' in s:
                        code.language = 'bash'
                return code
        else:
            if inside:
                if len(code.language) == 0 and (x == 'python' or x == 'bash'):
                    code.language = x
                else:
                    code.lines.append(x)

    return code


def execute_script(code):
    output = None
    err = None
    msg = None
    for i,x in enumerate(code.lines):
        k = i + 1
        console.print(f'{k:02d} {x}', style='red')
    # breakpoint()
    if code.language.lower() == 'python':
        output, err = save_and_execute_python(code)
        if err:
            msg = err
        else:
            msg = output
    elif code.language.lower() == 'bash':
        output, err = save_and_execute_bash(code)
        if err:
            msg = err
        else:
            msg = output
    else:
        console.print('unrecognised code block found')
    return msg


def create_client(llm_name):
    if llm_name == 'llama3':
        return OpenAI(api_key=os.environ['TOGETHERAI_API_KEY'], base_url='https://api.together.xyz/v1')
    elif llm_name == 'gpt35' or llm_name == 'gpt4':
        return OpenAI()
    elif llm_name == 'ollama':
        return OpenAI(api_key='dummy',  base_url="http://localhost:11434/v1")
    else: # lmstudio port
        return OpenAI(api_key='dummy', base_url='http://localhost:1234/v1')


def chat(llm_name):
    client = create_client(llm_name)
#    systemMessage = ChatMessage('system', FNCALL_SYSMSG)
    systemMessage = ChatMessage('system', f'You are Marvin a super intelligent AI chatbot trained by OpenAI. You are a logical thinker who answers concisely and accurately. The current datetime is {datetime.datetime.now().isoformat()}. you can write code in markdown code blocks.')
#    systemMessage = ChatMessage('system', f'You are Marvin a super intelligent AI chatbot trained by OpenAI. You are a logical thinker. The current datetime is {datetime.datetime.now().isoformat()}. You should use python to calculate mathematical expressions. Do not guess the output.')
    # systemMessage = ChatMessage('system', 'You are a loyal and dedicated member of the Koopa Troop, serving as an assistant to the infamous Bowser. You are committed to carrying out Bowsers commands with unwavering dedication and devotion. Lets work this out in a step by step way to make sure we have the right answer.')
    messages = [systemMessage]
    print(f'model={model_name.get(llm_name, "local")} . Enter x to exit.')
    inp = ''
    while (inp != 'x'):
        inp = input()
        if len(inp) > 3:
            msg = None
            if inp.startswith('%load'):
                msg = ChatMessage('user', load_msg(inp))
            elif inp.startswith('%resp'):
                msg = ChatMessage('user', tool_response(inp))
            elif inp.startswith('%reset'):
                messages.clear()
                messages.append(systemMessage)
                continue
            else:              
                msg = ChatMessage('user', inp)
            messages.append(msg)
            prt(msg)
            response = client.chat.completions.create(model=model_name.get(llm_name, 'local'), messages=[asdict(m) for m in messages], temperature=0.2)
            # response = client.chat.completions.create(model=model, messages=[asdict(m) for m in messages], tools=TOOL_FN, tool_choice="none", temperature=0.2)
            m = response.choices[0].message
            # if it is a tool call automatically reply and get the next reponse
            if m.tool_calls:
                r = process_tool_call(m.tool_calls[0])
                if r:
                    xs = [asdict(m) for m in messages]
                    xs.append(r)
                    response = client.chat.completions.create(model=model_name.get(llm_name, 'local'), messages=xs, tools=TOOL_FN, tool_choice="none", temperature=0.2)
#                    messages.append(ChatMessage('tool', str(r)))
                    m = response.choices[0].message
#            breakpoint()
            code = extract_code_block(m.content, '```')
            while code and len(code.language) > 0:
                # store original message from gpt
                msg = ChatMessage(m.role, m.content)
                messages.append(msg)
                prt(msg)
                output = execute_script(code)
                code = None
                if output:
                    msg2 = ChatMessage('user', '## output from running script\n' + output + '\n')
                    messages.append(msg2)
                    prt(msg2)
                    response = client.chat.completions.create(model=model_name.get(llm_name, ''), messages=[asdict(m) for m in messages], temperature=0.2)
#                    response = client.chat.completions.create(model=model_name.get(llm_name, ''), messages=[asdict(m) for m in messages], tools=TOOL_FN, tool_choice="auto", temperature=0.2)
                    m = response.choices[0].message
                    code = extract_code_block(m.content, '```')
            # store original message from gpt
            if m.content:
                msg = ChatMessage(m.role, m.content)
                messages.append(msg)
                prt(msg)
            x = response.usage.completion_tokens
            y = response.usage.prompt_tokens
            print(f'Completion tokens: {x}, prompt tokens: {y}, total tokens: {response.usage.total_tokens} cost: {(x * .2 + y * .1)/1000}')
            
    save(messages, make_fullpath(FNAME))


def chat_ollama():
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "magicoder:7b-s-cl-q6_K",
        "prompt":"Write a Python function to tell me what the date is today"
    }
    print('call endpoint')
    response = requests.post(url, data=json.dumps(data))
    resp = ''
    for s in response.text.split('\n'):
        r = json.loads(s)
        resp += r.get('response')
    return resp


def x():
    '''extract response field'''
    s = '{"model":"magicoder:7b-s-cl-q6_K","created_at":"2024-01-01T22:28:32.629495852Z","response":".","done":false}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat with LLMs')
    parser.add_argument('llm', type=str, help='The action to perform [local|gpt35|gpt4|llama3]')
    args = parser.parse_args()
    chat(args.llm)
    # s = chat_ollama()
    # print(s)
    # xs = load(FNAME)

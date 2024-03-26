#!/usr/bin/env python3
import argparse
from dataclasses import dataclass, asdict
from typing import List
from dataclasses_json import dataclass_json
from openai import OpenAI
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich import print as rprint
import requests
import json
import subprocess


# pip install dataclasses-json

FNAME = 'chat-log.json'
console = Console()

@dataclass_json
@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


def make_fullpath(fn: str) -> Path:
    return Path.home() / 'Documents' / fn


def save_and_execute_python(s: str) -> str:
    console.print('executing code...', style='red')
    full_path = make_fullpath('temp.py')
    with open(full_path, 'w') as f:
        f.write(s)
    
    result = subprocess.run(["python", full_path], capture_output=True)
    output = result.stdout.decode("utf-8")
    err = result.stderr.decode("utf-8")
    if len(output) > 0:
        rprint(Markdown(output))
    else:
        console.print(err, style='red')
#    console.print(output, style='green')
    return output, err


def prt(msg : ChatMessage):
    s = 'cyan' if msg.role == 'assistant' else 'yellow'
    console.print(msg.role + ': ', style=s)
#    print(msg.content)
    if msg.content.find('```') >= 0:
        md = Markdown(msg.content)
        console.print(md)
        xs = msg.content.split('```')
        if len(xs) > 0:
            x = xs[1]
            if x.startswith('python\n'):
                save_and_execute_python(x[7:])
            else:
                console.print('code block found but not python ' + x)
    else:
        console.print(msg.content, style=s)
    # print(colored(msg.role + ': ', c))
    # print(colored(msg.content, c))


def save(xs, filename):
    with open(filename, 'w') as f:
        f.write(ChatMessage.schema().dumps(xs, many=True))


def load_msg(s: str) -> str:
    xs = s.split()
    with open(make_fullpath(xs[1]), 'r') as f:
        return f.read()
    

def load(filename: str) -> List[ChatMessage]:
    with open(filename, 'r') as f:
        return ChatMessage.schema().parse_raw(f.read())


def chat(local=True, model=None):
    client = OpenAI(api_key="dummy") if local else OpenAI()
#    client = OpenAI()
    if local:
        client.base_url = 'http://localhost:1234/v1'
    systemMessage = ChatMessage('system', 'You are Marvin a helpful chatbot trained by OpenAI. You can use use python or Linux command line tools to complete your tasks. To use python or a command line tool place the code inside a markdown code block and then wait for the output.')
    # systemMessage = ChatMessage('system', 'You are a loyal and dedicated member of the Koopa Troop, serving as an assistant to the infamous Bowser. You are committed to carrying out Bowsers commands with unwavering dedication and devotion. Lets work this out in a step by step way to make sure we have the right answer.')
    messages = [systemMessage]
    print(f'model={model} . Enter x to exit.')
    inp = ''
    while (inp != 'x'):
        inp = input()
        if len(inp) > 3:
            msg = None
            if inp.startswith('%load'):
                msg = ChatMessage('user', load_msg(inp))
            else:              
                msg = ChatMessage('user', inp)
            messages.append(msg)
            prt(msg)
            response = client.chat.completions.create(model=model, messages=[asdict(m) for m in messages], temperature=0.2)

            m = response.choices[0].message
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
    parser.add_argument('llm', type=str, help='The action to perform [local|gpt35|gpt4]')
    args = parser.parse_args()
    if args.llm == 'gpt35':
        chat(False, model='gpt-3.5-turbo')
    elif args.llm == 'gpt4':
        chat(False, model='gpt-4-1106-preview')
    else:
        chat()
    # s = chat_ollama()
    # print(s)
    # xs = load(FNAME)

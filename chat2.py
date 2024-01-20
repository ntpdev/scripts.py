#!/usr/bin/env python3
import argparse
from dataclasses import dataclass, asdict
from typing import List
from dataclasses_json import dataclass_json
from openai import OpenAI
from pathlib import Path
from termcolor import colored

# pip install dataclasses-json

FNAME = 'chat-log.json'

@dataclass_json
@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


def make_fullpath(fn: str) -> Path:
    return Path.home() / 'Documents' / fn


def prt(msg : ChatMessage):
    c = 'cyan' if msg.role == 'assistant' else 'yellow'
    print(colored(msg.role + ': ', c))
    print(colored(msg.content, c))


def save(xs, filename):
    with open(filename, 'w') as f:
        f.write(ChatMessage.schema().dumps(xs, many=True))


def load(filename: str) -> List[ChatMessage]:
    with open(filename, 'r') as f:
        return ChatMessage.schema().parse_raw(f.read())


def chat(local=True, model='gpt-3.5-turbo'):   
    client = OpenAI()
    if local:
        client.base_url = 'http://localhost:1234/v1'
    # systemMessage = ChatMessage('system', 'You are an expert python programmer. Lets work this out in a step by step way to make sure we have the right answer.')
    systemMessage = ChatMessage('system', 'You are a loyal and dedicated member of the Koopa Troop, serving as an assistant to the infamous Bowser. You are committed to carrying out Bowsers commands with unwavering dedication and devotion. Lets work this out in a step by step way to make sure we have the right answer.')
    messages = [systemMessage]
    print(f'model={model} . Enter x to exit.')
    inp = ''
    while (inp != 'x'):
        inp = input()
        if len(inp) > 3:
            msg = ChatMessage('user', inp)
            messages.append(msg)
            prt(msg)
            response = client.chat.completions.create(model=model, messages=[asdict(m) for m in messages])

            m = response.choices[0].message
            msg = ChatMessage(m.role, m.content)
            messages.append(msg)
            prt(msg)
            x = response.usage.completion_tokens
            y = response.usage.prompt_tokens
            print(f'Completion tokens: {x}, prompt tokens: {y}, total tokens: {response.usage.total_tokens} cost: {(x * .2 + y * .1)/1000}')
            
    save(messages, make_fullpath(FNAME))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat with LLMs')
    parser.add_argument('llm', type=str, help='The action to perform [local|gpt35|gpt4]')
    args = parser.parse_args()
    if args.llm == 'gpt35':
        chat(False)
    elif args.llm == 'gpt4':
        chat(False, model='gpt-4-1106-preview')
    else:
        chat()
    # xs = load(FNAME)

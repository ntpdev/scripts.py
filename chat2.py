from dataclasses import dataclass, asdict
from typing import List
from dataclasses_json import dataclass_json
from openai import OpenAI
from termcolor import colored

# pip install dataclasses-json

FNAME = 'c:\\temp\\chat-log.json'

@dataclass_json
@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


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


def chat(local=True):   
    client = OpenAI()
    if local:
        client.base_url = 'http://localhost:1234/v1'
    systemMessage = ChatMessage('system', 'You are an expert programmer. Write concise python code. Add type hints in functions.')
    messages = [systemMessage]
    inp = ''
    while (inp != 'x'):
        inp = input()
        if len(inp) > 3:
            msg = ChatMessage('user', inp)
            messages.append(msg)
            prt(msg)
            response = client.chat.completions.create(model='gpt-3.5-turbo', messages=[asdict(m) for m in messages])

            m = response.choices[0].message
            msg = ChatMessage(m.role, m.content)
            messages.append(msg)
            prt(msg)
            x = response.usage.completion_tokens
            y = response.usage.prompt_tokens
            print(f'Completion tokens: {x}, prompt tokens: {y}, total tokens: {response.usage.total_tokens} cost: {(x * .2 + y * .1)/1000}')
            
    save(messages, FNAME)


if __name__ == '__main__':
    chat(False)
    # xs = load(FNAME)

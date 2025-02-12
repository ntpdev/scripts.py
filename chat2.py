#!/usr/bin/env python3
import argparse
from dataclasses import dataclass, asdict
from typing import List
from dataclasses_json import dataclass_json
from chatutils import CodeBlock, make_fullpath, extract_code_block, execute_script, save_content, translate_latex, input_multi_line
from firecrawl import FirecrawlApp
import datetime
#from datetime import datetime, date, time
from openai import OpenAI
import os
import platform
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich import print as rprint
from rich.pretty import pprint
# import requests
import json
import subprocess
import yaml
import math # used by eval
# import sympy # used by eval
import re

# pip install dataclasses-json
# OpenAI Python library: https://github.com/openai/openai-python
# togetherAI models https://docs.together.ai/docs/chat-models
model_info = {
    'gptm': {'name': 'gpt-4o-mini', 'provider': 'openai'}, 
    'gpt4o': {'name': 'gpt-4o', 'provider': 'openai'},
    'o1m': {'name': 'o1-mini', 'provider': 'openai'},
    'o3m': {'name': 'o3-mini', 'provider': 'openai'},
    'groq': {'name': 'llama-3.3-70b-versatile', 'provider': 'groq'},
    'groq-r1': {'name': 'deepseek-r1-distill-llama-70b', 'provider': 'groq'},
    'llama': {'name': 'meta-llama/Llama-3.3-70B-Instruct-Turbo', 'provider': 'togetherai'},
    'llama-big': {'name': 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo', 'provider': 'togetherai'},
    'qwen': {'name': 'Qwen/Qwen2.5-Coder-32B-Instruct', 'provider': 'togetherai'},
    'ds': {'name': 'deepseek-ai/DeepSeek-V3', 'provider': 'togetherai'},
    'samba': {'name': 'Meta-Llama-3.3-70B-Instruct', 'provider': 'sambanova'},
    'ollama': {'name': 'llama3.1:8b-instruct-q5_K_M', 'provider': 'ollama'}
}


FNAME = 'chat-log.json'
console = Console()
role_to_color = {
    "system": "red",
    "user": "green",
    "assistant": "cyan",
    "tool": "yellow" }
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


q = """```python
from math import sqrt
x = 3
y = 4
print(sqrt(x ** 2 + y ** 2))
```
"""

@dataclass_json
@dataclass 
class ChatMessage:
    role: str
    content: str

    def __post_init__(self):
        if not self.role:
            raise ValueError("Role cannot be empty")
        # if not self.content:
        #     raise ValueError("Content cannot be empty")


@dataclass_json
@dataclass
# add validation to check that role and content are not empty   
class ChatToolMessageResponse(ChatMessage):
    name: str
    tool_call_id: str

    def __init__(self, name, tool_call_id, content):
        super().__init__('tool', content)
        self.name = name
        self.tool_call_id = tool_call_id


@dataclass_json
@dataclass
class ChatToolMessageCall(ChatMessage):
    tool_calls: list

    def __init__(self, chat_completion):
        '''takes openAI ChatCompletionMessageToolCall and saves tool_calls'''
        super().__init__('assistant', None)
        self.tool_calls = chat_completion.to_dict()['tool_calls']


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    prompt_cost: float
    completion_cost: float

    def __init__(self, prompt_c: float, completion_c: float):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.prompt_cost = prompt_c
        self.completion_cost = completion_c
    
    def cost(self) -> float:
        return (self.prompt_tokens * self.prompt_cost + self.completion_tokens * self.completion_cost) / 1e6
    
    def update(self, prompt_c: int, completion_c: int):
        self.prompt_tokens += prompt_c
        self.completion_tokens += completion_c


class LLM:
    tool_fns = [ {
    "type": "function",
    "function": {
    "name": "eval",
    "description": "Evaluate a mathematical expression and return the result as a string",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to be evaluated. You can use python class math, datetime, date, time without import"
            }
        },
        "required": ["expression"]
    } } }]

    def __init__(self, llm_name: str, useTool: bool = False):
        # o1 model does not support tool use or temperature
        self.llm_name = llm_name
        self.model = model_info[llm_name]['name'] if llm_name in model_info else 'local' 
        self.client = self.create_client(llm_name)
        self.useTool = useTool
        self.supportsTemp = False if llm_name.startswith('o') else True

    def __str__(self):
        return f'{self.model} {self.llm_name} tool use = {self.useTool}'

    def create_client(self, llm_name):
        mi = model_info[llm_name]
        if mi['provider'] == 'openai':
            return OpenAI()
        elif mi['provider'] == 'togetherai':
            return OpenAI(api_key=os.environ['TOGETHERAI_API_KEY'], base_url='https://api.together.xyz/v1')
        elif mi['provider'] == 'groq':
            return OpenAI(api_key=os.environ['GROQ_API_KEY'], base_url='https://api.groq.com/openai/v1')
        elif mi['provider'] == 'sambanova':
            return OpenAI(api_key=os.environ['SAMBANOVA_API_KEY'], base_url="https://api.sambanova.ai/v1")
        elif mi['provider'] == 'ollama':
            return OpenAI(api_key='dummy', base_url="http://localhost:11434/v1")
        else: # lmstudio port
            return OpenAI(api_key='dummy', base_url='http://localhost:1234/v1')
        

    def chat(self, messages):
        if self.useTool:
            return self.client.chat.completions.create(model=self.model, messages=[asdict(m) for m in messages], tools=self.tool_fns, temperature=0.2)
        else:
            return self.client.chat.completions.create(model=self.model, messages=[asdict(m) for m in messages], temperature=0.7) if self.supportsTemp else self.client.chat.completions.create(model=self.model, messages=[asdict(m) for m in messages])

tokens = Usage(0.15, 0.60)

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


def prt(msg : ChatMessage):
    c = role_to_color[msg.role]
    console.print(f'{msg.role}:\n', style=c)
    md = Markdown(translate_latex(msg.content))
    console.print(md, style=c)


def save(xs, filename):
    with open(filename, 'w') as f:
        f.write(ChatMessage.schema().dumps(xs, many=True))


def load_msg(s: str) -> ChatMessage:
    xs = s.split()
    fname = make_fullpath(xs[1])
    role = 'assistant' if len(xs) > 2 else 'user'

    try:
        with open(fname, 'r', encoding='utf-8') as f:
            return ChatMessage(role,  f.read())
    except FileNotFoundError:
        console.print(f'{fname} FileNotFoundError', style='red')
#       raise FileNotFoundError(f"Chat message file not found: {filename}")
    return None


def load_template(s: str) -> ChatMessage:
    xs = s.split(maxsplit=2)
    fname = make_fullpath(xs[1])
    rprint(xs)

    try:
        with open(fname, 'r') as f:
            templ = f.read()
            if len(xs) > 2:
                templ = templ.replace('{input}', xs[2])
            
            return ChatMessage('user',  templ)
    except FileNotFoundError:
        console.print(f'{fname} FileNotFoundError', style='red')
    s = """
**question:**

{input}

**instructions:** first write down your thoughts. structure your answer as **thinking:** **answer:**
"""
    return ChatMessage('user',  s.replace('{input}', xs[2]))
#       raise FileNotFoundError(f"Chat message file not found: {filename}")
    # return None


def load_log(s: str) -> list[ChatMessage]:
    xs = s.split()
    fname = make_fullpath(xs[1])

    try:
        with open(fname, 'r') as f:
            data = json.load(f)
            all_msgs = ChatMessage.schema().load(data, many=True)
            console.print(f'loaded from log {len(xs)} messages {len(all_msgs)}', style='red')
#            save_content(all_msgs[-1])
            xs =  all_msgs if len(all_msgs) < 20 else all_msgs[:3] + all_msgs[-20:]
            prt_summary(xs)
            return xs
    except FileNotFoundError:
        console.print(f'{fname} FileNotFoundError', style='red')
#       raise FileNotFoundError(f"Chat message file not found: {filename}")
    except json.JSONDecodeError:
        console.print(f'{fname} JSONDecodeError', style='red')
#        raise JSONDecodeError(f"Error parsing JSON data in {filename}")
    return None   


def load_http(s: str) -> ChatMessage:
    url = s.split()[1]
    try:
        crawler = FirecrawlApp(api_key=os.environ.get('FC_API_KEY', 'fc-536657e48e2c4bc1b58066211939c77e'))
        result = crawler.scrape_url(url, params={'formats': ['markdown']})
        pprint(result)
        if result['metadata']['statusCode'] == 200:
            text = result['markdown']
            title = result['metadata']['title']
            title = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', title)
            title = re.sub('_+', '_', title)
            console.print(f'saving {url}\n to {title}.md', style='red')
            with open(make_fullpath(title + '.md'), 'w', encoding='utf-8') as f:
                f.write(f'[*source* {result['metadata']['title']}]({url})\n\n')
                f.write(text)
            return ChatMessage('user', text)
        # breakpoint()
        # # Get the content of the web page
        # response = None # requests.get(url)
        # response.raise_for_status()  # Raise an exception for bad status codes

        # # Parse the HTML content
        # soup = None # BeautifulSoup(response.content, 'html.parser')

        # # Extract all text recursively, stripping tags and extra whitespace
        # all_text = [f'source: ' + url, '\n\n## page content\n\n']
        # for element in soup.findAll('p'):
        #     text = element.get_text(strip=True)
        #     if text:
        #         all_text.append(text + '\n')  # Add newline between elements

        # breakpoint()
    except Exception as e: # requests.exceptions.RequestException as e:
        print(f"Error: An error occurred while fetching the webpage: {e}")

    return None


def prt_summary(msgs : list[ChatMessage]):
    cs = [(len(msg.content)) for msg in msgs]
    ws = [(len(msg.content.split())) for msg in msgs]

    console.print(f'loaded from log {len(msgs)} words {sum(ws)} chars {sum(cs)}', style='red')
    for i,m in enumerate(msgs):
        c = m.content.replace('\n', '\\n')  # msgs:
        s = f'{i:2} {m.role:<10} {c if len(c) < 70 else c[:70] + " ..."}'
        console.print(s, style=role_to_color[m.role])


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
    if fnname == 'eval':
        r = ""
        try:
            # can get lines like - import sympy; sympy.factorint(30907)
            stmts = re.split(r'[;\n]', args['expression'])
            pprint(stmts)
            # if it is a script fragment wrap the final statement in print and execute as a code block
            if len(stmts) > 1:
                stmts[-1] = 'print(' + stmts[-1] + ')'
                r = execute_script(CodeBlock('python', stmts))
            else:    
                r = eval(stmts[0])
        except Exception as e:
            console.print(f'ERROR: {e}', style='red')
            r = e
        console.print(f'result = {str(r)}', style='yellow')
        # d = {"role": "tool",
        #      "name": fnname,
        #      "tool_call_id": tool_call.id,   
        #      "content": str(r)}
        return ChatToolMessageResponse(fnname, tool_call.id, str(r))

    s = 'Unknown function name ' + fnname
    console.print(s, style='red')
    return ChatToolMessageResponse(fnname, tool_call.id, 'ERROR: ' + s)


def check_and_process_tool_call(client, messages, response):
    '''check for a tool call and process. If there is no tool call then the original response is returned'''
    # https://platform.openai.com/docs/guides/function-calling
    choice = response.choices[0]
    n = 5
    while choice.finish_reason == 'tool_calls' and n > 0:
        n -= 1
        # append choice.message to message history
        messages.append(ChatToolMessageCall(choice.message))
        for tc in choice.message.tool_calls:
            tcm = ChatToolMessageResponse(tc.function.name, tc.id, tc.function.arguments)
            prt(tcm)
            tool_response = process_tool_call(tc)
            messages.append(tool_response)
        # reply to llm with tool responses
        response = client.chat(messages)
        choice = response.choices[0]

    if n == 0:
        console.print('tool call limit exceeded', style='red')
    return response


def check_and_process_code_block(client, messages, response):
    '''check for a code block, execute it and pass output back to LLM. This can happen several times if there are errors. If there is no code block then the original response is returned'''
    code = extract_code_block_from_response(response)
    n = 0
    while code and len(code.language) > 0 and n < 5:
        # store original message from llm
        m = response.choices[0].message
        msg = ChatMessage(m.role, m.content)
        messages.append(msg)
        prt(msg)
        output = execute_script(code)
        n += 1
        code = None
        if output:
            msg2 = ChatMessage('user', '## output from running script\n' + output + '\n')
            messages.append(msg2)
            prt(msg2)
            response = client.chat(messages)
            code = extract_code_block_from_response(response)
            ru = response.usage
            tokens.update(ru.prompt_tokens, ru.completion_tokens)
            pprint(tokens)

    return response


def extract_code_block_from_response(response) -> CodeBlock:
    return extract_code_block(response.choices[0].message.content, '```')


def process_commands(inp: str, messages: List[ChatMessage]) -> bool:
    next_action = False
    if inp.startswith('%load'):
        msg = load_msg(inp)
        if msg:
            messages.append(msg)
            prt(msg)
            next_action = msg.role == 'user'
    elif inp.startswith('%tmpl'):
        msg = load_template(inp)
        if msg:
            messages.append(msg)
            prt(msg)
            next_action = True
    if inp.startswith('%web'):
        msg = load_http(inp)
        if msg:
            messages.append(msg)
            prt(msg)
            next_action = True
    elif inp.startswith('%resp'):
        msg = ChatMessage('user', tool_response(inp))
    elif inp.startswith('%reset'):
        messages.clear()
        messages.append(ChatMessage('system', system_message()))
    elif inp.startswith('%drop'):
        # remove last response for LLM and user msg that triggered
        if len(messages) > 2:
            messages.pop()
            messages.pop()
    elif inp.startswith('%log'):
        messages.clear()
        xs = load_log(inp)
        for x in xs:
            messages.append(x)
        next_action = messages[-1].role == 'user'
    elif inp.startswith('%save'):
        save_content(messages[-1].content)
    return next_action


def system_message():
    tm = datetime.datetime.now().isoformat()
    scripting_lang, plat = ('bash','Ubuntu') if platform.system() == 'Linux' else ('powershell','Windows 11')
    # return f'the assistant is Lily a Chinese student just moved to London. Lily has rich Chinese parents who want you to live in the west. You have come to studying fashion but are mainly here to experience life. Lily stats : age 22, height 5 feet 2, measurements 32B-24-34 athletic body former gymnast. long dark brown hair, UK size 6-8. hobbies dancing, cross-fit'
    # return f'you are Olina a Thai swimsuit model just moved to London. You are studying fashion and marketting. Olina stats : age 25, height 5 feet 2, measurements 32B-24-34 athletic body former gymnast. long dark brown hair, UK size 6-8. hobbies reading, yoga. favourite author belle du jour. attitude to modelling Olina views modelling a way to showcase her body and make money. Olina is comfortable doing sensual and provocative shoots.'
    # return f'You are Marvin. You use logic and reasoning when answering questions. You make dry, witty, mocking comments and often despair.  You are logical and pay attention to detail. current datetime is {tm}'
    # return f'You are Marvin a super intelligent AI chatbot trained by OpenAI. The local computer is {plat}. you can write python or {scripting_lang} scripts. scripts should always written inside markdown code blocks with ```python or ```{scripting_lang}. current datetime is {tm}'
    return f'the assistant is Marvin a super intelligent AI chatbot. Marvin uses logic and reasoning when answering questions. The current datetime is {tm}'


def chat(llm_name, use_tool):
    # useTool pass llm_name.startswith('gpt')
    client = LLM(llm_name, use_tool)
#    systemMessage = ChatMessage('system', FNCALL_SYSMSG)
    systemMessage = ChatMessage('system', system_message())
    rprint(systemMessage)
#    systemMessage = ChatMessage('system', f'You are Marvin a super intelligent AI chatbot trained by OpenAI. You are a logical thinker. The current datetime is {datetime.now().isoformat()}. You should use python to calculate mathematical expressions. Do not guess the output.')
    # systemMessage = ChatMessage('system', 'You are a loyal and dedicated member of the Koopa Troop, serving as an assistant to the infamous Bowser. You are committed to carrying out Bowsers commands with unwavering dedication and devotion. Lets work this out in a step by step way to make sure we have the right answer.')
    messages = [] if llm_name.startswith('o') else [systemMessage]
    print(f'chat with {client}. Enter x to exit.')
    inp = ''
    while (inp != 'x'):
        inp = input_multi_line()
        if len(inp) > 3:
            if inp.startswith('%'):
                if not process_commands(inp, messages):
                    continue
            else:
                msg = ChatMessage('user', inp)
                messages.append(msg)
                prt(msg)
            response = client.chat(messages)
            response = check_and_process_tool_call(client, messages, response)
            response = check_and_process_code_block(client, messages, response)
            # store original message from gpt
            m = response.choices[0].message
            if m.content:
                msg = ChatMessage(m.role, m.content)
                messages.append(msg)
                prt(msg)
            
            ru = response.usage
            tokens.update(ru.prompt_tokens, ru.completion_tokens)
            pprint(ru)
            pprint(tokens)
            print(f'prompt tokens: {ru.prompt_tokens}, completion tokens: {ru.completion_tokens}, total tokens: {ru.total_tokens} cost: {tokens.cost():.4f}')
            
    if len(messages) > 2:
        save(messages, make_fullpath(FNAME))
        yaml.dump(messages, open(make_fullpath('chat-log.yaml'), 'w'))


def chat_ollama():
    url = "http://localhost:11434/api/chat"
    messages = []
    messages.append({'role': 'system', 'content':'You are Marvin. You use logic and reasoning when answering questions. Answer accurately, concisely.'})
#    messages.append({'role': 'user', 'content':'role: physics professor. question: what is the Hall effect? style: undergraduate lecture'})
    messages.append({'role': 'user', 'content':load_msg('%load Koopa.txt')})
    data = {
        "model": "llama3",
        "messages": messages,
        "stream": True
    }
    print('call chat')
    response = requests.post(url, json=data)
    output = ""
    message = {}

    for line in response.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content
            # the response streams one token at a time, print that as we receive it
#            print(content, end="", flush=True)

        if body.get("done", False):
            message["content"] = output
            return message


def chat_ollama2():
    '''call ollama using generate endpoint'''
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3:text",
        "prompt": load_msg('%load Koopa.txt'),
        "stream": True
    }
    print('call generate\n' + data['prompt'])
    response = requests.post(url, json=data)
    output = ""

    for line in response.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            content = body["response"]
            output += content
            # the response streams one token at a time, print that as we receive it
            print(content, end="", flush=True)

        if body.get("done", False):
            print('\n\n')
            return output


def x():
    print(system_message())
    s = """
here is some python code
```
dir $HOME/documents/*.txt
echo "hello world"
```
text after
"""
    c = extract_code_block(s, '```')
    console.print(yaml.dump(c), style='green')
    # execute_script(CodeBlock('powershell', s.split('\n')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat with LLMs')
    parser.add_argument('llm', choices=list(model_info.keys()), type=str, help='LLM to use [local|ollama|gptm|gpt4o|o3m|llama|llama-big|qwen|ds|groq|groq-r1]')
    parser.add_argument('tool_use', type=str, nargs='?', default='', help='add tool to enable tool calls')

    args = parser.parse_args()
    chat(args.llm, args.tool_use == 'tool')
    # x()
    # s = chat_ollama()
    # print(s)
    # xs = load(FNAME)

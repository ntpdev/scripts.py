import argparse
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from datetime import datetime, date, time
from rich.console import Console
from rich.markdown import Markdown
from rich import print as rprint
from chatutils import CodeBlock, make_fullpath, extract_code_block, execute_script
import subprocess
import os
from pathlib import Path
from textwrap import dedent

# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/configure-safety-attributes
safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

# tool_fns = [{
#     "type": "function",
#     "function": {
#         "name": "eval",
#         "description": "Evaluate a mathematical expression and return the result as a string",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "expression": {
#                     "type": "string",
#                     "description": "The mathematical expression to be evaluated. You can use python class math, datetime, date, time without import"
#                     }
#                 },
#             "required": ["expression"]
#             }
#         }
#     }]

console = Console()
#llm2 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
#llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.2)
#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, base_url='http://localhost:1234/v1')


@tool
def tool_eval(expression: str) -> str:
    """Evaluate a mathematical expression. The expression can include use python class math, datetime"""
    console.print('eval: ' + expression, style='yellow')
    r = ""
    try:
        r = eval(expression)
        console.print('result: ' + str(r), style='yellow')
    except Exception as e:
        r = 'ERROR: ' + str(e)
        console.print(r, style='red')
    return r


def process_tool_calls(messages):
    '''process tool calls and add results to message history'''
    for call in messages[-1].tool_calls:
        if call['name'].lower() == 'tool_eval':
            r = tool_eval(call['args']['expression'])
            messages.append(ToolMessage(r, tool_call_id = call['id']))
        else:
            messages.append(ToolMessage('unknown tool: ' + call['name'], tool_call_id = call['id']))


def check_and_process_code_block(llm, messages, max_executions):
    '''check for a code block, execute it and pass output back to LLM. This can happen several times if there are errors. If there is no code block then the original response is returned'''
    aimsg = messages[-1]
    code = extract_code_block(aimsg.content, '```')
    n = 0
    while code and len(code.language) > 0 and n < max_executions:
        # print original message from llm
        output = execute_script(code)
        n += 1
        code = None
        if output:
            msg = HumanMessage('## output from running script\n' + output + '\n')
            messages.append(msg)
            print_message(msg)
            aimsg = llm.invoke(messages)
            messages.append(aimsg)
            print_message(aimsg)
            code = extract_code_block(aimsg.content,  '```')


def print_message(m):
    c = 'cyan'
    role = 'assistant'
    if isinstance(m, SystemMessage):
        c = 'red'
        role = 'system'
    elif isinstance(m, HumanMessage):
        c = 'green'
        role = 'user'
    elif isinstance(m, ToolMessage):
        c = 'yellow'
        role = 'tool'
    
    console.print(f'\n{role}:', style=c)
    if m.content:
        md = Markdown(m.content)
        console.print(md, style=c)
    elif len(m.tool_calls) > 0:
        console.print(m.tool_calls[0], style=c)


def print_history(messages):
    for m in messages:
        print_message(m)


messages = [
    SystemMessage(f'The current datetime is {datetime.now().isoformat()}.'),
    HumanMessage(dedent("""
use python to print the contents of ~\\Documents\\q2.md               
                        """))
]

def create_llm(llm_name, toolUse):
    if llm_name == 'pro':
        llm = ChatVertexAI(model='gemini-1.5-pro-preview-0514',  safety_settings=safety_settings)
    elif llm_name == 'haiku':
        llm = ChatAnthropicVertex(model_name='claude-3-haiku')
    else:
        llm = ChatVertexAI(model='gemini-1.5-flash-preview-0514',  safety_settings=safety_settings)
    if toolUse and llm_name != 'haiku':
        llm = llm.bind_tools([tool_eval])
    return llm


def chat(llm_name):
    llm = create_llm(llm_name, False)
    console.print('chat with model: ' + llm.model_name, style='yellow')
    msg = llm.invoke(messages)

    while len(msg.tool_calls) > 0:
        messages.append(msg)
        print_message(msg)
        process_tool_calls(messages)
        breakpoint()
        msg = llm.invoke(messages)

    messages.append(msg)
    print_history(messages)

    check_and_process_code_block(llm, messages, 3)

    inp = ''
    while (inp != 'x'):
        inp = input()
        if len(inp) > 3:
            messages.append(HumanMessage(inp))
            msg = llm.invoke(messages)
            messages.append(msg)
            print_message(msg)
            

    # execute_code_reply(messages, 3)

#    print_history(messages)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat with LLMs')
    parser.add_argument('llm', type=str, help='LLM to use [flash|pro|haiku]')
    args = parser.parse_args()
    chat(args.llm)
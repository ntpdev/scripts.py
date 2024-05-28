import argparse
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from datetime import datetime, date, time
import platform
import math
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
store = {}
#llm2 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
#llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.2)
#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, base_url='http://localhost:1234/v1')


@tool
def tool_eval(expression: str) -> str:
    """Evaluate a mathematical expression. The expression can include use python class math, datetime"""
    #console.print('eval: ' + expression, style='yellow')
    r = ""
    try:
        exp = expression.replace('datetime.datetime', 'datetime')
        exp = expression.replace('datetime.date', 'date')
        console.print('eval: ' + exp, style='yellow')
        r = eval(exp)
        console.print('result: ' + str(r), style='yellow')
    except Exception as e:
        r = 'ERROR: ' + str(e)
        console.print(r, style='red')
    return r


def process_tool_calls(msg):
    '''process tool calls and add return ToolMessages'''
    messages = []
    for call in msg.tool_calls:
        if call['name'].lower() == 'tool_eval':
            r = tool_eval(call['args']['expression'])
            messages.append(ToolMessage(r, tool_call_id = call['id']))
        else:
            messages.append(ToolMessage('unknown tool: ' + call['name'], tool_call_id = call['id']))
    return messages


def check_and_process_tool_calls(llm, history):
    msg = history.messages[-1]
    while len(msg.tool_calls) > 0:
        toolmsgs = process_tool_calls(msg)
        history.add_messages(toolmsgs)
        msg = llm.invoke(history.messages)
        history.add_message(msg)
        print_message(msg)


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


def load_msg(s: str) -> BaseMessage:
    xs = s.split()
    fname = make_fullpath(xs[1])
    is_human = len(xs) <= 2
    
    try:
        with open(fname, 'r') as f:
            s = f.read()
            return HumanMessage(s) if is_human else AIMessage(s)
    except FileNotFoundError:
        console.print(f'{fname} FileNotFoundError', style='red')
#       raise FileNotFoundError(f"Chat message file not found: {filename}")
    return None


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def single_message(llm):
    prompt = ChatPromptTemplate.from_messages([system_message(), load_msg('%load q2.md')])
    h = get_session_history('z1')
    h.add_message(system_message())
    chain = prompt | llm
    chain_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="question", history_messages_key="history")
    m = chain_history.invoke({}, config={'configurable': {'session_id': 'z1'}})
    rprint(store)
    # rprint(get_session_history('z1'))
    

def system_message():
    tm = datetime.now().isoformat()
    scripting_lang, plat = ('bash','Ubuntu 23.10') if platform.system() == 'Linux' else ('powershell','Windows 11')
#    return f'You are Marvin a super intelligent AI chatbot trained by OpenAI. You use deductive reasoning to answer questions. You make dry, witty, mocking comments and often despair.  You are logical and pay attention to detail. You can access local computer running {plat} by writing python or {scripting_lang}. Scripts should always be in markdown code blocks with the language. current datetime is {tm}'
    return SystemMessage(f'You are Marvin a super intelligent AI chatbot. The local computer is {plat}. you can write python or {scripting_lang} scripts. scripts should always written inside markdown code blocks with ```python or ```{scripting_lang}. current datetime is {tm}')


# messages = [
#     system_message(),
# #     HumanMessage(dedent("""
# # hello. what is the time.              
# #                         """))
# ]

def create_llm(llm_name, temp, toolUse):
    if llm_name == 'pro':
        llm = ChatVertexAI(model='gemini-1.5-pro-preview-0514',  safety_settings=safety_settings, temperature=temp)
    elif llm_name == 'haiku':
        llm = ChatAnthropicVertex(model_name='claude-3-haiku', temperature=temp)
    else:
        llm = ChatVertexAI(model='gemini-1.5-flash-preview-0514',  safety_settings=safety_settings, temperature=temp)
    if toolUse and llm_name != 'haiku':
        llm = llm.bind_tools([tool_eval])
    return llm


def chat(llm_name):
    llm = create_llm(llm_name, 0.2, True)
    console.print('chat with model: ' + llm.model_name, style='yellow')
    history = ChatMessageHistory()
    history.add_message(system_message())

    inp = ''
    while (inp != 'x'):
        inp = input().strip()
        msg = None
        if len(inp) > 3:
            if inp.startswith('%load'):
                msg = load_msg(inp)
                if msg is None:
                    continue
                elif isinstance(msg, AIMessage):
                    history.add_message(msg)
                    print_message(msg)
                    continue
            else:
                msg = HumanMessage(inp)
            print_message(msg)
            history.add_message(msg)
            msg = llm.invoke(history.messages)
            history.add_message(msg)
            # messages.append(msg)
            print_message(msg)
            check_and_process_tool_calls(llm, history)
            # check_and_process_code_block(llm, messages, 3)
            
    print_history(history.messages)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat with LLMs')
    parser.add_argument('llm', type=str, help='LLM to use [flash|pro|haiku]')
    args = parser.parse_args()
    chat(args.llm)
import random
from . import prompts as pt
from types import FunctionType

REPLY_RECUSION_LIMIT = 5

class Roles:
    '''Namespace'''
    
    @staticmethod
    def user(msg):
        return {'role': 'user', 'content': msg}
    
    @staticmethod
    def system(msg):
        return {'role': 'system', 'content': msg}
    
    @staticmethod
    def assistant(msg):
        return {'role': 'assistant', 'content': msg}
    
roles = Roles()


def parse_command(command):
    split = command.replace('(',' ').replace(')',' ').replace("\"", '').split()
    tool_name = split.pop(0)
    args = []
    kwargs = {}
    for s in split:
        if '=' in s:
            kwargs[s.split('=')[0].strip()] = s.split('=')[1].strip()
        else:
            args.append(s)
    return tool_name, args, kwargs


def doc_string(obj):
    if hasattr(obj, '_specialisation_msg'):
        return f'name: {obj.name} \nusage: {obj._specialisation_msg}\n'
    elif isinstance(obj, FunctionType):
        return f'name: {obj.__name__} \nusage: {obj.__doc__ }\n'
    else:
        return repr(obj)


def doc_string_ai(doc_str):
    '''
    Use an LLM to check doc string, rewrite if needed.
    '''
    # LATER: ^
    return doc_str

def remove_tool_from_prompt(prompt, tool_name, tools):
    tool_names = [t for t in tools]
    n_tools_remaining = len(set(tool_names) - {tool_name})
    prompt[0]['content'] = prompt[0]['content'].replace(doc_string(tools[tool_name]), '') 
    if n_tools_remaining == 0:
        prompt[0]['content'] = prompt[0]['content'].replace(pt.DEFAULT_TOOL_PROMPT, '')


def auto_name():
    return 'chatllm-' + str(random.randint(1e5, 1e6))

def is_toolcall(response):
    return '<C@LL>' in response

def is_agentcall(response):
    return ('<C0NS#LT>' in response) or ('<RELE@SE>' in response)

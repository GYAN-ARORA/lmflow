from typing import List
from copy import deepcopy
from abc import ABC, abstractmethod
from lmflow import prompts as pt
from lmflow import util
from lmflow.util import roles
from lmflow.closed_source import activate_agent, execute_tool

class ChatLM(ABC):
    '''Conversation manager for your desired chat completion LLM API multi-agent support'''

    def __init__(self, name=None, system_prompt=pt.DEFAULT_SYSTEM_PROMPT):
        self.__user_commands = system_prompt
        self.__tool_prompt = ''
        self.__agent_prompt = ''
        self._specialisation_msg = ''
        self._control_path = ['__self__']
        self._system_prompt = roles.system(self.__user_commands)
        self.prompt = [self._system_prompt]
        self.tools = {}
        self.agents = {}
        self.name = util.auto_name() if name is None else name
        
    @property
    def _in_control(self):
        return self if self._control_path[-1] == '__self__' else self._control_path[-1]
        
    def __execute_tool(self, response, prompt):
        # Not making this function public yet
        return execute_tool(self, response, prompt)
    
    def __activate_agent(self, response):
        # Not making this function public yet
        return activate_agent(self, response)

    def __respond(self, prompt: List[str]) -> str:
        '''Enforces validity of system prompt'''
        # TODO: Handle in case there are two consecutive user msgs, delete the last one (happens when a command errors out)
        prompt[0] = self._system_prompt
        return self.respond(prompt)
    
    def __get_response(self, prompt):
        response = self._in_control.__respond(prompt)
        if util.is_toolcall(response): # TODO: Add this back to the recurssion
            response = self._in_control.__execute_tool(response, prompt)
        return response
    
    def __reply(self) -> str:
        for _ in range(util.REPLY_RECUSION_LIMIT):
            response = self._in_control.__get_response(self.prompt)
            if util.is_agentcall(response):
                self.__activate_agent(response)
            else:
                return response
        raise RecursionError("Recursion limit reached before a final reply could be generated")
    
    @abstractmethod
    def respond(self, prompt: List[str]) -> str:
        '''Stateless API call to your desired LLM. Dont keep any state here'''
        pass
    
    def reply(self, msg: str) -> str:
        self.prompt.append(roles.user(msg))
        response = self.__reply()
        self.prompt.append(roles.assistant(response))
        return response

    def clear_chat(self):
        self.instruct()
        self.prompt = [self._system_prompt]

    def clear_last_interaction(self):
        self.prompt = self.prompt[:-2]
    
    def instruct(self, msg: str = '', overwrite=False) -> None:
        if overwrite:
            self.__user_commands = msg
        else:
            self.__user_commands += msg
        self._system_prompt = roles.system(self.__user_commands + '\n' +  
                                           self.__tool_prompt + '\n' + 
                                           self.__agent_prompt)
        self.prompt[0] = self._system_prompt

    def add_tools(self, tools):
        if not isinstance(tools, list):
            tools = [tools]
        for tool in tools:
            if tool.__qualname__.split('.')[0] != 'lmtool':
                raise TypeError(f'function {tool.__name__} is not a valid tool. Please use lmflow.base.lmtool decorator with it.')
        for tool in tools:
            self.tools[tool.__name__] = tool
        self.__tool_prompt = pt.DEFAULT_TOOL_PROMPT + '\n'.join([util.doc_string(self.tools[name]) for name in self.tools])
        self.instruct()

    def add_agents(self, agents):
        if not isinstance(agents, list):
            agents = [agents]
        for agent in agents:
            if not isinstance(agent, ChatLM):
                raise TypeError("agent should be an instance of class lmflow.base.ChatLM")
            if not agent._specialisation_msg:
                raise ValueError(f"agent {agent} does not have a usage information. Please set it using set_user_guide() method")
        for agent in agents:
            self.agents[agent.name] = agent
        self.__agent_prompt = pt.DEFAULT_AGENT_PROMPT + '\n'.join([util.doc_string(self.agents[name]) for name in self.agents])
        self.instruct()

    def set_specialisation(self, specialisation_msg):
        '''Set the task that this agent will be used for by the parent agent. Format - "You specialise in ..."'''
        if not specialisation_msg.startswith("This agent specialises in "):
            raise ValueError('Specialisation mesage must always start with the phrase - "This agent specialises in "')
        self._specialisation_msg = specialisation_msg.lower()
        specialisation_msg = specialisation_msg.replace('this agent', 'you').replace('specialises', 'specialise')
        self.instruct(pt.DEFAULT_AGENT_SYSTEM_PROMPT.replace('<PURP05E_PL@CE#0LDER>', specialisation_msg), overwrite=True)
    
    def save(self):
        # think of this again after multi conversation support is added
        raise NotImplementedError
    
    def load(self):
        raise NotImplementedError
    

def lmtool(human=False, dont_infer=False):
    def inner_tool(func):
        if not func.__doc__:
            raise ValueError('Tool definitions must contain a standard doc string to tell the AI how to use it')
        def inner(*args, **kwargs):
            if human:
                resp = input(f'Please confirm the following action (Y/N): {func.__name__} {args} {kwargs}').lower()
                if not (resp == 'y' or resp == 'yes'):
                    return "Task cancelled by user", True
            output = func(*args, **kwargs)
            output = '' if output is None else output
            return str(output), dont_infer
        inner.__doc__ = util.doc_string_ai(func.__doc__)
        inner.__name__ = func.__name__
        return inner
    return inner_tool
from copy import deepcopy
from lmflow import util
from lmflow import prompts as pt

def activate_agent(self, response):
    if '<C0NS#LT>' in response:
        agent_name = response.strip('<C0NS#LT>').strip(' ').split(' ')[0]
        if agent_name not in self._in_control.agents: # LATER: add a retry mechanism
            raise KeyError(f"Assistant '{self._in_control.name}' tried to transfer to an agent: '{agent_name}' which does not exist")
        print('\033[95m' + f'Transfering conversation to agent: "{agent_name}"' + '\033[0m')
        self._control_path.append(self._in_control.agents[agent_name])
    elif '<RELE@SE>' in response:
        self._control_path.pop()
        print('\033[95m' + f'Transfering conversation back to agent: "{self._in_control.name}"' + '\033[0m')
    return None

def execute_tool(self, response, prompt):
        tool_name, args, kwargs = util.parse_command(response.strip('<C@LL>').strip(' '))
        if tool_name not in self.tools: # LATER: add a retry mechanism
            raise KeyError(f"Assistant '{self.name}' tried to call a tool: '{tool_name}' which does not exist")
        output, dont_infer = self.tools[tool_name](*args, **kwargs)
        if dont_infer or (not output):
            return f'Task Completed!\n{output}'
        response = self.tools[tool_name].__doc__.split('\n')[0] + ' :\n' + output # highly depends on doc string
        temp_prompt = deepcopy(prompt)
        temp_prompt[0] = self._system_prompt
        temp_prompt[-1]['content'] = pt.DEFAULT_TOOL_REPLY + response + '\n\n' + temp_prompt[-1]['content']
        util.remove_tool_from_prompt(temp_prompt, tool_name, self.tools) # remove use of the same tool again
        response = self.respond(temp_prompt)
        return response

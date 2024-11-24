DEFAULT_SYSTEM_PROMPT = '''
You are a helpful assistant. Keep your responses concise and do not provide elaborations unless requested.
'''

DEFAULT_AGENT_SYSTEM_PROMPT = f'''{DEFAULT_SYSTEM_PROMPT}\n <PURP05E_PL@CE#0LDER>\n
You should only answer queries related to this subject. \
For any query not related to this subject reply with the keyword <RELE@SE>".
'''

DEFAULT_TOOL_PROMPT = '''
You have certain functions in the following list to help you. Only if any of these functions can help with \
the query, respond with the keyword <C@LL> followed by the function name and parameters required to be \
passed to it. Do not use the <C@LL> keyword for any other fuction or purpose.

'''


DEFAULT_AGENT_PROMPT = '''
You have certain agents in the following list to help you which specialise in certain areas. Only if any \
of these agents can help with the query, respond with the keyword <C0NS#LT> followed by the agent name. \
Do not use the <C0NS#LT> keyword for any other purpose.

'''

DEFAULT_TOOL_REPLY = 'use this information to answer the following question\n\n'

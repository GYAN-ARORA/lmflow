# LMFLow

A simple, clean python framework for building multi-agent applications powered by your favourite LLMs


## Usage

### ChatLM 
```python
# create a model inference class

from lmflow.base import ChatLM

class MyLLM(ChatLM):

    def __init__(model_name=..., **kwargs):
        self.model = openAI(model_name, **kwargs)
        super.__init__()

    def respond(prompt: List[str]);
        resp = self.model.invoke(prompt)
        return resp['response']


llm = MyLLM(...)
llm.predict(prompt='...')
```

The `ChatLM` base class abstracts away whatever underlying API you want to use and provides some useful features such as a `reply` function which takes care of the message history automatically. So you can just send the current user message to it for a response. It works under the hood as follows
```python
def reply(msg: str)
    self.msg_history.append(msg)
    reply_msg = self.respond(self.msg_history)
    self.msg_history.append(reply_msg)
    print(reply_msg)
```
Other features include, token counting, cashing, persisting, etc.

This version only focuses on chat completion (i.e. `QA` or `Instruct`) models. However, other common NLP tasks maybe supported in the future. The framework however does not restrict custom classes and usage.
Some popular LLMs are already available out of the box like 

```python
from lmflow.LM import (
    OpenAIChatLLM,
    AnthropicChatLM,
    HuggingFaceLM
)
```

### Tools

Note than tools can only input data types that can be converted from strings safely and can only return str

```python
from lmflow.base import lmtool

@lmtool
def journeys(passenger_id: int, source_id: str=None):
    """
    provide <passenger_id> and <source_id> to get booked journeys
    """
    ...
    return journey_dataframe

llm.add_tools(journies)
llm.add_tools([name_to_id, todays_date, cancel_ticket])

llm.reply("Cancel Gyan Arora's all upcoming journeys from Bengaluru for next 2 months")
```
```
> Canceled journey from Bengaluru to Surat on 23rd Oct 2024, PNR: QYG765. Do you need any further assistance ?
```

#### Human in the loop

A human confirmation for any tool can be added as follows. Msg customisation is possible as an advanced feature and requires access to decision loop.

```python
@lmtool(human=True)
def cancel_ticket(PNR: str):
    ...
    return 

llm.add_tool(cancel_ticket)
llm.reply("Cancel Gyan Arora's all upcoming journeys from Bengaluru for next 2 months")
```
```
> Please confirm the following action - Cancel journey from Bengaluru to Surat on 23rd Oct 2024, PNR: QYG765.

Yes / Any other instruction (no) ? 
```

#### Inspect how this happened

```python
llm.msg_history[-10:]
```

### Multi-Agent Flow
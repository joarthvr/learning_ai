from typing import Annotated, TypedDict

from langgraph.graph import add_messages


class MyDate(TypedDict):
    messages: Annotated[list, add_messages]  #! Reducer


from langchain_core.messages import AIMessage, HumanMessage

msg1 = [HumanMessage(content='안녕', id='1')]
msg2 = [AIMessage(content='방가', id='2')]

results = add_messages(msg1, msg2)

# print(results)
# ? [HumanMessage(content='안녕', additional_kwargs={}, response_metadata={}, id='1'),
# ? HumanMessage(content='방가', additional_kwargs={}, response_metadata={}, id='2')]

# --------------------------------------------------------------------------------

msg3 = [HumanMessage(content='안녕', id='1')]
msg4 = [AIMessage(content='방가', id='1')]

results2 = add_messages(msg3, msg4)
print(results2)
# ? [AIMessage(content='방가', additional_kwargs={}, response_metadata={}, id='1', tool_calls=[], invalid_tool_calls=[])]

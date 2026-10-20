#! tool calling: 외부 기능을 가져와서 사용
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import SecretStr

load_dotenv()

DB_PATH = './_db/Chroma_rag11/'
BASE_URL = 'https://monogpt.kr/api/monorouter/v1/'
API_KEY = SecretStr(os.environ['MONOROUTER_API_KEY'])


llm = ChatOpenAI(
    model='gpt-6-luna',
    temperature=0,
    model_kwargs={'max_tokens': 1000},
    api_key=API_KEY,
    base_url=BASE_URL,
)
#! 0. Tool 정의 -----------------------------------------------------

# 웹 검색 도구 생성

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# DuckDuckGoSearchRun은 직접 max_results를 받지 않고, 내부 API wrapper를 통해 설정합니다.
ddg_wrapper = DuckDuckGoSearchAPIWrapper()
web_search = DuckDuckGoSearchRun(api_wrapper=ddg_wrapper)

tools = [web_search]

llm_with_tools = llm.bind_tools(tools)


#! 1. State 정의 -----------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


#! 2. Node 정의 ------------------------------------------------------


def chatbot(state: State):
    return {
        'messages': [
            llm_with_tools.invoke(state['messages']),
        ]
    }


#! 3. 그래프 구성, 노드 추가 --------------------------------------------

# 그래프 생성
graph_builder = StateGraph(State)

graph_builder.add_node('chatbot', chatbot)

tool_node = ToolNode(tools)
graph_builder.add_node('tools', ToolNode(tools))


#! 4. 엣지 추가 -------------------------------------------------------
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_conditional_edges(
    'chatbot',
    tools_condition,  # ? tools call 이 있나?
    # ? tool_calls가 있으면 'tools' 반환
    # ? tool_calls가 없으면 END 반환
    # ? tools_condition에 END 포함돼있음
)
graph_builder.add_edge('tools', 'chatbot')
# graph_builder.add_edge('chatbot', END) #? 따라서 엔드를 명시할 필요가 없음


#! 5. 그래프 컴파일 ----------------------------------------------------
graph = graph_builder.compile()

#! 6. 그래프 실행 ------------------------------------------------------
query = '삼성전자의 주가를 예측해줘'
res = graph.invoke(
    {
        'messages': [('user', query)],
    }
)


#! 7. 결과 출력 ---------------------------------------------------------
print('AI: ', res['messages'][-1].content)
"""
for i, msg in enumerate(res['messages']):
    print(f'[{i}] {msg.type} {msg.content}')
"""

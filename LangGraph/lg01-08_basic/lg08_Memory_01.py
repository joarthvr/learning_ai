import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
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

web_search = TavilySearch(
    max_result=3,
)
"""
result = web_search.invoke(
    {
        'query': '삼성전자의 오늘 주가 얼마',
    }
)

print(result['results'][-1]['content'])
"""

tools = [web_search]

llm_with_tools = llm.bind_tools(tools)

from typing import Annotated, TypedDict

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


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
res = graph.invoke(
    {
        'messages': [
            ('user', '내 이름은 윤영선'),
        ]
    }
)

print(res['messages'][-1].content)

print('------------------------ 두번째 질문 ----------------------------')

res = graph.invoke(
    {
        'messages': [
            ('user', '내 이름이 뭐라고 했지?'),
        ]
    }
)
print(res['messages'][-1].content)

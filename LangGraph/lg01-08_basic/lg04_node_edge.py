import os
from typing import Annotated, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
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


#! 1. State 정의 -----------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    aaa: NotRequired[str]
    bbb: NotRequired[str]
    ccc: NotRequired[int]


#! 2. Node 정의 -----------------------------------------------------


def prepare(state: State):
    return {
        'messages': [
            AIMessage(
                content='질문을 접수했습니다.',
            )
        ],
    }


def chatbot(state: State):
    return {
        'messages': [
            llm.invoke(state['messages']),
        ]
    }


def finish(state: State):
    return {
        'messages': [
            AIMessage(
                content='답변 처리가 완료됐습니다.',
            )
        ],
    }


#! 3. 그래프 구성 -----------------------------------------------------
graph_01 = StateGraph(State)

graph_01.add_node('prepare', prepare)
graph_01.add_node('chatbot', chatbot)
graph_01.add_node('finish', finish)

#! 4. 엣지 추가 -------------------------------------------------------
graph_01.add_edge(START, 'prepare')
graph_01.add_edge('prepare', 'chatbot')
graph_01.add_edge('chatbot', 'finish')
graph_01.add_edge('finish', END)


#! 5. 그래프 컴파일 ----------------------------------------------------
graph = graph_01.compile()

#! 6. 그래프 실행 ------------------------------------------------------
query = '삼성전자의 주가를 예측해줘'
res = graph.invoke(
    {
        'messages': [('user', query)],
    }
)

print(res)

#! 7. 결과 출력 ---------------------------------------------------------
for i, msg in enumerate(res['messages']):
    print(f'[{i}] {msg.type} {msg.content}')

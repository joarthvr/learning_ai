import os
from io import BytesIO
from typing import Annotated, NotRequired, TypedDict

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from PIL import Image
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

tools = [web_search]

llm_with_tools = llm.bind_tools(tools)


#! 1. State 정의 -----------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    search_result: NotRequired[str]
    analysis_result: NotRequired[str]
    final_result: NotRequired[str]


#! 2. Node 정의 ------------------------------------------------------
#! 검색 에이전트
def search_agent(state: State):
    print('검색 에이전트 실행')
    question = state['messages'][-1].content

    result = web_search.invoke(
        {
            'query': question,
        }
    )
    return {'search_result': str(result)}


def analysis_agent(state: State):
    print('분석 에이전트 실행')
    question = state['messages'][-1].content

    res = llm.invoke(f'다음 질문을 기존 지식을 바탕으로 분석해줘. \n{question}')
    return {'analysis_result': res.content}


def merge_agent(state: State):
    result = (
        state.get('search_result', '')
        + '\n\n'
        + '-----------------------------------------'
        + '\n'
        + state.get('analysis_result', '')
    )
    return {'final_result': result}


def chatbot(state: State):
    return {
        'messages': [
            llm_with_tools.invoke(state['messages']),
        ]
    }


#! 3. 그래프 구성 --------------------------------------------
builder = StateGraph(State)
builder.add_node('search_agent', search_agent)
builder.add_node('analysis_agent', analysis_agent)
builder.add_node('merge_agent', merge_agent)


#! 4. 병렬 실행 --------------------------------------------
builder.add_edge(START, 'search_agent')
builder.add_edge(START, 'analysis_agent')

#! 5. 병합 --------------------------------------------
builder.add_edge('search_agent', 'merge_agent')
builder.add_edge('analysis_agent', 'merge_agent')

builder.add_edge('merge_agent', END)

#! 6. 컴파일 --------------------------------------------
graph = builder.compile()
# ------------------------- 그래프 실행 -------------------------------

#! 7. 실행 --------------------------------------------
query = '엔비디아의 미래 전망에 대해 알려줘'

res = graph.invoke({'messages': [('user', query)]})

print('Assistant:', res['messages'][-1].content)

#! 8. 그래프 이미지 저장 ----------------------------------------------------


PATH = './_save/image/'
png_data = graph.get_graph().draw_mermaid_png()

img = Image.open(BytesIO(png_data))

# 그림 출력
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.show()

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


#! 1. State 정의 -----------------------------------------------------
class State(TypedDict):
    question: str
    search_result: str
    analysis: str
    final_result: str


#! 2. 검색 Node 정의 ------------------------------------------------------
def search_agent(state: State):
    print('검색 에이전트 실행')
    return {'search_result': '엔비디아 AI GPU 시장의 주요 기업이다.'}


#! 3. 분석 노드 추가 --------------------------------------------
def analysis_agent(state: State):
    print('분석 에이전트 실행')
    return {'analysis': 'AI 반도체 수요 증가가 성장 요인이다.'}


#! 4. 결과 통합 Node --------------------------------------------
def merge_agent(state: State):
    result = state['search_result'] + '\n' + state['analysis']
    return {'final_result': result}


#! 5. 그래프 구성 --------------------------------------------
builder = StateGraph(State)
builder.add_node('search_agent', search_agent)
builder.add_node('analysis_agent', analysis_agent)
builder.add_node('merge_agent', merge_agent)


#! 6. 병렬 실행 --------------------------------------------
builder.add_edge(START, 'search_agent')
builder.add_edge(START, 'analysis_agent')

#! 7. 병합 --------------------------------------------
builder.add_edge('search_agent', 'merge_agent')
builder.add_edge('analysis_agent', 'merge_agent')

builder.add_edge('merge_agent', END)

#! 8. 컴파일 --------------------------------------------
graph = builder.compile()


#! 8. 실행 --------------------------------------------
initial_state: State = {
    'question': '엔비디아의 미래 전망은?',
    'search_result': '',
    'analysis': '',
    'final_result': '',
}
res = graph.invoke(initial_state)

print(res['final_result'])


#! 9. 그래프 이미지 저장 ----------------------------------------------------

from io import BytesIO

import matplotlib.pyplot as plt
from PIL import Image

PATH = './_save/image/'
png_data = graph.get_graph().draw_mermaid_png()

img = Image.open(BytesIO(png_data))

# 그림 출력
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.show()

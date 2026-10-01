# [학습 정리] rag01_basic 02 api_key 인자 생략하기
# - ChatOpenAI 는 api_key 를 안 주면 환경변수 OPENAI_API_KEY 를 알아서 찾는다
# - 그래서 load_dotenv() 로 환경변수에만 올려두면 01 처럼 직접 넘길 필요가 없다

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
os.environ[
    'OPENAI_API_KEY'
]  # 시스템 환경변수  # → 값을 쓰지 않는 줄. 키가 없으면 KeyError 로 먼저 멈추는 확인용

llm = ChatOpenAI(
    model='gpt-5.6-terra',
    temperature=0,
    # api_key=SecretStr(os.environ['OPENAI_API_KEY']),
)
response = llm.invoke('안녕하세요')
print(response)
print(response.content)

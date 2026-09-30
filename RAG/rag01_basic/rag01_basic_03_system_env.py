# [학습 정리] rag01_basic 03 .env 없이 OS 시스템 환경변수로만 키 쓰기
# - load_dotenv() 를 주석 처리 → 윈도우 '시스템 환경 변수' 에 OPENAI_API_KEY 가 등록돼 있어야 동작
# - 환경변수를 새로 등록했다면 VS Code/터미널을 재시작해야 반영된다 (실행 중인 프로세스는 옛 값을 가짐)

import os

# from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# load_dotenv()
os.environ['OPENAI_API_KEY']  # 시스템 환경변수

llm = ChatOpenAI(
    model='gpt-5.6-terra',
    temperature=0,
    # api_key=SecretStr(os.environ['OPENAI_API_KEY']),
)
response = llm.invoke('안녕하세요')
print(response)
print(response.content)

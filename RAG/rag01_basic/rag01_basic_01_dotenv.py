# [학습 정리] rag01_basic 01 API 키로 LLM 한 번 호출하기
# - 키를 코드에 직접 쓰면 깃에 올라가는 순간 유출 → .env 파일에 두고 load_dotenv() 로 읽는다 (.env 는 gitignore)
# - SecretStr 로 감싸면 print/log 에 키가 '**********' 로 가려진다
# - invoke() 의 반환값은 AIMessage 객체 → 실제 답변 텍스트는 .content

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()  # 현재 파일 위치부터 상위 폴더로 올라가며 .env 를 찾아 os.environ 에 넣는다

llm = ChatOpenAI(
    model='gpt-5.6-terra',
    temperature=0,  # 0 = 가장 결정적 (같은 질문 → 거의 같은 답)
    api_key=SecretStr(os.environ['OPENAI_API_KEY']),
)
response = llm.invoke('안녕하세요')
print(response)  # 토큰 사용량 등 메타데이터까지 전부
print(response.content)

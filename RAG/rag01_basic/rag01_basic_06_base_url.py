# [학습 정리] rag01_basic 06 base_url 로 다른 API 서버 쓰기
# - OpenAI 호환 API 를 제공하는 서비스는 base_url 만 바꾸면 ChatOpenAI 코드를 그대로 쓸 수 있다
# - 키도 서비스별로 따로 → .env 에 MONOROUTER_API_KEY 로 분리해서 관리

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

api_key = SecretStr(os.environ['MONOROUTER_API_KEY'])
base_url = 'https://monogpt.kr/api/monorouter/v1/'

llm = ChatOpenAI(
    model='gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)
response = llm.invoke('안녕')
# print(response)
print(response.content)

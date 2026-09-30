# [학습 정리] rag01_basic 05 temperature 바꿔보기 (01 복사)
# - temperature 는 다음 토큰을 고를 때의 무작위성. 0 = 가장 확률 높은 토큰만, 높을수록 다양하고 엉뚱해진다
# - 2 는 OpenAI 기준 최댓값 → 같은 질문에도 매번 답이 크게 달라지고 문장이 깨질 수 있다

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

llm = ChatOpenAI(
    model='gpt-5.6-terra',
    temperature=2,  # 01_dotenv 의 0 과 비교해 보기
    api_key=SecretStr(os.environ['OPENAI_API_KEY']),
)
response = llm.invoke('안녕')
print(response)
print(response.content)

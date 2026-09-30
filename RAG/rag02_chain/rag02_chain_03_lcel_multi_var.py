import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

api_key = SecretStr(os.environ['MONOROUTER_API_KEY'])
base_url = 'https://monogpt.kr/api/monorouter/v1/'

model = ChatOpenAI(
    model='gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

prompt = PromptTemplate.from_template('{topic}에 대해 쉽게 {how} 설명해주세요.')
chain = prompt | model

user_input = {'topic': '양자컴퓨터 학습 원리', 'how': '초등학생도 이해하기 쉽게'}

res = chain.invoke(user_input)
print(res.content)

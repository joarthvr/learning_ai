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

prompt = PromptTemplate.from_template('{topic}에 대해 쉽게 설명해주세요.')
# ======================================================================

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
# ======================================================================

chain = prompt | model | output_parser

user_input = {'topic': '양자컴퓨터 학습 원리'}

res = chain.invoke(user_input)
print(res)

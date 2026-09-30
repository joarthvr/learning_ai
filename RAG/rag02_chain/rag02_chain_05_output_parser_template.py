import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

template = """
당신은 영어를 가르치는 10년차 영어 선생님입니다.
주어진 상황에 맞는 영어 회화에 맞는 영어회화를 작성해 주세요.
양식은 [FORMAT]을 참고하여 작성해 주세요.

#상황:
{question}

#FORMAT:
- 영어회화 :
- 한글번역 :
"""

load_dotenv()

api_key = SecretStr(os.environ['MONOROUTER_API_KEY'])
base_url = 'https://monogpt.kr/api/monorouter/v1/'

model = ChatOpenAI(
    model='gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

prompt = PromptTemplate.from_template(template=template)
# ======================================================================

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
# ======================================================================

chain = prompt | model | output_parser

user_input = {'question': '저는 부산에서 물밀면을 먹고싶어요.'}

res = chain.invoke(user_input)
print(res)

import os

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

api_key = SecretStr(os.environ['MONOROUTER_API_KEY'])
base_url = 'https://monogpt.kr/api/monorouter/v1/'

prompt = '삼성전자의 창업주는 누구인가요?'

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
)

vector = embeddings.embed_query(prompt)
print(vector)
print(len(vector))  # 1536

# rag03_Embedding_01.py 카피
import os

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

api_key = SecretStr(os.environ['MONOROUTER_API_KEY'])
base_url = 'https://monogpt.kr/api/monorouter/v1/'

prompt = '삼성전자의 창업주는 누구인가요?'

#! 허깅페이스 임베딩 ------------------------------------------------------------------------

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model='BAAI/bge-m3',
    model_kwargs={
        'device': 'cuda',
        # 'local_files_only' : True,
    },
)

vector = embeddings.embed_query(prompt)
print('임베딩 벡터의 차원: ', len(vector))

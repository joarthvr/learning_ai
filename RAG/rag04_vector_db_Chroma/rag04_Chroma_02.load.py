import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import SecretStr

load_dotenv()

api_key = SecretStr(os.environ['MONOROUTER_API_KEY'])
base_url = 'https://monogpt.kr/api/monorouter/v1/'

path = './_data/rag_data/'
loader1 = TextLoader(path + 'samsung_outlook.txt', encoding='utf-8')  # 한글문서 인코딩
loader2 = TextLoader(path + 'nvidia_outlook.txt', encoding='utf-8')


#! 임베딩 ----------------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
)

DB_PATH = './_db/Chroma11/'
db = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma11',
)

#! 저장된 데이터 확인 ------------------------------------------------------------------------
print(db.get())

aaa = db.similarity_search('삼성전자 사업전망에 대해서 알려줘', k=2)
print(aaa)

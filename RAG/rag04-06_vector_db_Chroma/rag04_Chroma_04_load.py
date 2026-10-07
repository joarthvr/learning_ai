import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import SecretStr

load_dotenv()

DB_PATH = './_db/Chroma12/'
api_key = SecretStr(os.environ['MONOROUTER_API_KEY'])
base_url = 'https://monogpt.kr/api/monorouter/v1/'


#! 3. 임베딩 --------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
)

#! LOAD ---------------------------------------------------------------------------------
db = Chroma(
    embedding_function=embeddings,  # 어떤 임베딩인지 알려줘야함
    persist_directory=DB_PATH,
    collection_name='chroma12',
)

query = '삼성전자의 창업자는 누구인가요?'
result = db.similarity_search(query)

#! --------------------------------- Retrievers (검색기) ----------------------------------
retriever = db.as_retriever(search_kwargs={'k': 2})
print(retriever)
# ? tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.
# ? Chroma object at 0x000001D12C2BDC50> search_kwargs={'k': 2}
aaa = retriever.invoke(query)
print(f'검색된 관련 문서 수 : {len(aaa)}')  # ? 검색된 관련 문서 수 : 2
print(f'{aaa[0].page_content[:50]}...')

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

texts = [
    Document(page_content='삼성전자는 반도체를 생산하는 기업이다.'),
    Document(page_content='삼성전자는 갤럭시 스마트폰을 판매한다.'),
    Document(page_content='SK하이닉스는 메모리 반도체를 생산한다.'),
    Document(page_content='엔비디아는 인공지능 연산용 GPU를 설계한다.'),
    Document(page_content='애플은 아이폰을 개발하는 기업이다.'),
    Document(page_content='테슬라는 전기자동차를 제조한다.'),
    Document(page_content='삼성전자는 AI 반도체 기술을 연구한다.'),
    Document(page_content='반도체는 컴퓨터의 핵심 부품이다.'),
]

query = '삼성전자의 반도체 사업'

load_dotenv()
DB_PATH = './_db/Chroma_rag11/'
BASE_URL = 'https://monogpt.kr/api/monorouter/v1/'
API_KEY = SecretStr(os.environ['MONOROUTER_API_KEY'])
#! ------------------------------------------- BM25 ---------------------------------------------

from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 3

#! ----------------------------------- vector search -------------------------------------------

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=API_KEY,
    base_url=BASE_URL,
)
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma_rag11',
)

vector_retriever = vector_store.as_retriever(search_kwargs={'k': 3})


print('--------------------- 벡터 검색 결과 -----------------------')

from langchain_classic.retrievers import EnsembleRetriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        vector_retriever,
    ],
    weights=[0.5, 0.5],
)

hybrid_docs = hybrid_retriever.invoke(query)

for docs in hybrid_docs:
    print(docs.page_content)

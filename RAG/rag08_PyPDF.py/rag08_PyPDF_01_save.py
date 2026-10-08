import os

import faiss
from dotenv import load_dotenv
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

load_dotenv()

PATH = './_data/'
DB_PATH = './_db/FAISS19/'
BASE_URL = 'https://monogpt.kr/api/monorouter/v1/'
API_KEY = SecretStr(os.environ['MONOROUTER_API_KEY'])


#! 1. 데이터 불러오기  --------------------------------------------------------------
pdf_loader = PyPDFLoader(PATH + 'attention_is_all_you_need.pdf')
pdf_docs = pdf_loader.load()
# print(len(pdf_docs))  # ? 15
# print(type(pdf_docs))  # ? <class 'list'>

#! 2. 문서를 자른다 / 청킹 ----------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=['\n\n', '\n', ' ', ''],
)

split_doc1 = text_splitter.split_documents(pdf_docs)  # 청크 300, 오버랩 100

print(len(split_doc1))  # ? 9 9

#! 3. 임베딩 ------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=API_KEY,
    base_url=BASE_URL,
)
#! --------------------------------- FAISS ----------------------------------
faiss_index = faiss.IndexFlatL2(
    len(embeddings.embed_query('hello world'))
)  # 유클리드 거리로 인덱스 검색
# faiss_index = faiss.IndexFlatL2(1536)
print('FAISS 인덱스 초기화 버튼 준비 완료')


# FAISS 벡터 저장소의 벡터 차원 수 (임베딩 차원 수)
print(faiss_index.d)  # 1536

faiss_db = FAISS(
    embedding_function=embeddings,
    index=faiss_index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# 저장된 문서의 개수 확인
print(faiss_db.index.ntotal)  # 0

db = FAISS.from_documents(
    documents=split_doc1,
    embedding=embeddings,
)

db.save_local(
    folder_path=DB_PATH,
    index_name='faiss_index19',
)

print('FAISS 문서저장 완료')

# rag04_Chroma_01_save 카피

import os

import faiss
from dotenv import load_dotenv
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

load_dotenv()

api_key = SecretStr(os.environ['MONOROUTER_API_KEY'])
base_url = 'https://monogpt.kr/api/monorouter/v1/'

path = './_data/rag_data/'
DB_PATH = './_db/FAISS17/'

loader1 = TextLoader(path + 'samsung_outlook.txt', encoding='utf-8')  # 한글문서 인코딩
loader2 = TextLoader(path + 'nvidia_outlook.txt', encoding='utf-8')

#! 1. 데이터 불러오기  --------------------------------------------------------------
split_doc1 = loader1.load_and_split()

#! 2. 문서를 자른다 / 청킹 ----------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    separators=['\n\n', '\n', ' ', ''],
)

split_doc1 = loader1.load_and_split(text_splitter)  # 청크 300, 오버랩 100
split_doc2 = loader2.load_and_split(text_splitter)  # 청크 300, 오버랩 100

print(len(split_doc1), len(split_doc2))  # ? 9 9

#! 3. 임베딩 ------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
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
    documents=split_doc1 + split_doc2,
    embedding=embeddings,
)

db.save_local(
    folder_path=DB_PATH,
    index_name='faiss_index17',
)

print('FAISS 문서저장 완료')

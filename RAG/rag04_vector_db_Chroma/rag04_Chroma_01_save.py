import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

load_dotenv()

api_key = SecretStr(os.environ['MONOROUTER_API_KEY'])
base_url = 'https://monogpt.kr/api/monorouter/v1/'

path = './_data/rag_data/'
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

# print(split_doc1, split_doc2)
print(len(split_doc1), len(split_doc2))  # ? 9 9

#! 3. 임베딩 ------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
)

DB_PATH = './_db/Chroma11/'
db = Chroma.from_documents(
    documents=split_doc1 + split_doc2,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma11',
)

print('Chroma 문서저장 완료')

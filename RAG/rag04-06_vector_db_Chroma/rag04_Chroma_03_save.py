import os
from glob import glob

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
txt_files = glob(os.path.join(path, '*.txt'))

# print(txt_files)
# ? ['./_data/rag_data\\2026_AI_for_All.txt',
# ? './_data/rag_data\\nvidia_outlook.txt',
# ? './_data/rag_data\\samsung_outlook.txt']

#! 1. 데이터 불러오기  --------------------------------------------------------------
documents = []
for txt_file in txt_files:
    loader = TextLoader(txt_file, encoding='utf-8')
    documents.extend(loader.load())

print(f'로드된 문서 수: {len(documents)}')  # ? 3
print('doc', documents)

char_count = [len(doc.page_content) for doc in documents]

print('char_count', char_count)
# ? char_count [8158, 2049, 1898]


#! 2. 문서를 자른다 / 청킹 ----------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    # separators=['\n\n', '\n', ' ', ''], # default 값
)

texts = text_splitter.split_documents(documents)

print('생성된 텍스트 청크 수 :', len(texts))
# ? 생성된 텍스트 청크 수 : 59

print('각 청크의 길이 :', list(len(text.page_content) for text in texts))
# ? 각 청크의 길이 : [259, 154, 150, 282, 128, 276, 288, 258, 268, 262, 271,
# ? 226, 268, 182, 213, 281, 257, 182, 162, 208, 259, 188, 225, 229, 219, 214,
# ? 258, 207, 284, 297, 198, 245, 177, 272, 215, 9, 269, 293, 290, 236, 289, 209,
# ? 222, 230, 254, 249, 296, 181, 247, 243, 185, 219, 239, 235, 298, 299, 170, 187, 249]

print('첫번째 청크의 내용 :', texts[0])
# ?  page_content='2026년 한국의 AI for All 프로젝트와 생성형 AI 서비스 확산
# ? 작성 목적
# ? 이 문서는 2026년 10월 초 공개된 국내 인공지능 관련 최신 보도를 바탕으로
# ? LangChain의 문서 로딩, 텍스트 분할, 임베딩, 벡터 데이터베이스 저장, 검색 및 RAG 실습에활용할 수 있도록 재구성한 학습용 텍스트이다.
# ? 특정 언론사의 기사 원문을 복제하지않고 공개된 사실과 기술적 배경을 중심으로 내용을 확장해 작성하였다.
# ? 1. 한국의 AI for All 프로젝트' metadata={'source': './_data/rag_data\\2026_AI_for_All.txt'}
# ! page_content, metadata가 들어있다

print('첫번째 청크의 길이 :', len(texts[0].page_content))
# ? 첫번째 청크의 길이 : 259

#! 3. 임베딩 --------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
)

sample_text = '삼성전자의 창업자는 누구인가요?'

query_vector = embeddings.embed_query(sample_text)
DB_PATH = './_db/Chroma12/'
db = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma12',
)
print(f'Chroma 문서저장 완료: {db._collection.count()}')
# ? Chroma 문서저장 완료: 118

query = '삼성전자의 창업자는 누구인가요?'
result = db.similarity_search(query)

print(f'검색 결과의 길이: {len(result)}')
# ? 검색 결과의 길이: 4

#! --------------------------------- Retrievers (검색기) ----------------------------------
retriever = db.as_retriever(search_kwargs={'k': 2})
print(retriever)
# ? tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.
# ? Chroma object at 0x000001D12C2BDC50> search_kwargs={'k': 2}
aaa = retriever.invoke(query)
print(f'검색된 관련 문서 수 : {len(aaa)}')  # ? 검색된 관련 문서 수 : 2

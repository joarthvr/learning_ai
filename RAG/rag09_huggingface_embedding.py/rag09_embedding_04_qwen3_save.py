from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PATH = './_data/'
DB_PATH = './_db/FAISS20/'
INDEX_NAME = 'faiss_index20'

#! 1. 데이터 불러오기  --------------------------------------------------------------
pdf_loader = PyPDFLoader(PATH + 'attention_is_all_you_need.pdf')
pdf_docs = pdf_loader.load()

#! 2. 문서를 자른다 / 청킹 ----------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=['\n\n', '\n', ' ', ''],
)
split_docs = text_splitter.split_documents(pdf_docs)
print(len(split_docs))

#! 3. 임베딩 (Qwen3, 로컬 실행 / 챗봇과 같은 모델이어야 한다) ------------------------
embeddings = HuggingFaceEmbeddings(
    model='Qwen/Qwen3-Embedding-0.6B',
    model_kwargs={'device': 'cpu'},
)

#! 4. FAISS 저장 -------------------------------------------------------------------
db = FAISS.from_documents(
    documents=split_docs,
    embedding=embeddings,
)
print(db.index.d)  # 1024
print(db.index.ntotal)

db.save_local(
    folder_path=DB_PATH,
    index_name=INDEX_NAME,
)

print('FAISS 문서저장 완료')

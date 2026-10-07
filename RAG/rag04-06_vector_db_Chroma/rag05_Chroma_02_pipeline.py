import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
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

query = '삼성의 파운드리 산업의 현재 현황은?'
result = db.similarity_search(query)

#! --------------------------------- Retrievers (검색기) ----------------------------------
retriever = db.as_retriever(search_kwargs={'k': 2})
print('------------------------------------------------------------')

#! --------------------------------- 모델 연결 ---------------------------------------------
model = ChatOpenAI(
    model='gpt-6-luna',
    temperature=0,
    max_completion_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """
    다음 컨텍스트를 바탕으로 질문에 답변해주세요. 컨텍스트 관련 정보가 없다면,
    "주어진 정보로는 답변할 수 없습니다."라고 말씀해주세요.

    컨텍스트: {context}

    질문: {input}

    답변:
    """
)


#! 체인 만들기 ---------------------------------------------------------------------------------
doc_chain = create_stuff_documents_chain(model, prompt)  # prompt | model
rag_chain = create_retrieval_chain(retriever, doc_chain)  # 검색 | doc_chain

#! 체인 실행 ---------------------------------------------------------------------------------
query = '삼성전자의 창업자는 누구인가요?'
res = rag_chain.invoke({'input': query})

print(res.keys())  # dict_keys(['input', 'context', 'answer'])
print(res['context'][0].page_content)
print(res['answer'])

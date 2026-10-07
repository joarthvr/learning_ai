# rag04_Chroma_01_save 카피

import os

import gradio as gr
from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import SecretStr

load_dotenv()

api_key = SecretStr(os.environ['MONOROUTER_API_KEY'])
base_url = 'https://monogpt.kr/api/monorouter/v1/'

path = './_data/rag_data/'
DB_PATH = './_db/FAISS17/'

#! 3. 임베딩 ------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
)
#! --------------------------------- FAISS ----------------------------------------------

#! LOAD ---------------------------------------------------------------------------------
db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index17',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

#! 유사도 검색 -----------------------------------------------------------------------------
query = '삼성의 파운드리 산업의 현재 현황은?'
result = db.similarity_search(query)

#! --------------------------------- 모델 연결 ---------------------------------------------
model = ChatOpenAI(
    model='gpt-6-luna',
    temperature=0,
    max_completion_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

#! --------------------------------- Retrievers (검색기) ----------------------------------
prompt = ChatPromptTemplate.from_template(
    """
    # 역할
    당신은 삼성전자 관련 자료를 분석하는 전문가입니다.

    # 지침
    * 아래 [컨텍스트]에 있는 내용만 근거로 답변합니다. 컨텍스트에 없는 내용은 추측하거나 지어내지 않습니다.
    * 컨텍스트에서 답을 찾을 수 없으면 "주어진 정보로는 답변할 수 없습니다."라고만 답합니다.
    * 결론을 첫 문장에 쓰고, 이어서 근거를 컨텍스트에서 인용해 설명합니다 (두괄식).
    * 수치, 날짜, 고유명사는 컨텍스트에 적힌 그대로 씁니다.
    * 불필요한 부사, 형용사, 대명사와 은유, 비유를 쓰지 않고 직접적이고 정확한 표현을 사용합니다.
    * 한국어로 답변합니다.
    * 질문자가 이해하기 쉽게 예를 들 수 있습니다.

    # 컨텍스트
    {context}

    # 질문
    {input}

    # 답변
    """
)

retriever = db.as_retriever(search_kwargs={'k': 4})

#! 체인 만들기 ---------------------------------------------------------------------------------
doc_chain = create_stuff_documents_chain(model, prompt)  # prompt | model
rag_chain = create_retrieval_chain(retriever, doc_chain)  # 검색 | doc_chain


#! --------------------------------- Gradio 챗봇 ---------------------------------------------
def answer_invoke(message, history):
    res = rag_chain.invoke({'input': message})
    return res['answer']


demo = gr.ChatInterface(fn=answer_invoke, title='chatbot_with_FAISS')

demo.launch(share=True)

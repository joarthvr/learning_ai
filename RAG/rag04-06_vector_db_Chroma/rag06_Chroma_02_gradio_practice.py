import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
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


prompt = ChatPromptTemplate.from_template(
    """
    당신은 삼성전자관련 전문가입니다.
    다음 컨텍스트를 바탕으로 질문에 답변해주세요. 컨텍스트 관련 정보가 없다면,
    "주어진 정보로는 답변할 수 없습니다."라고 말씀해주세요.
    관련된 자료와 근거를 토대로 인용하여 두괄식구조로 답변해주세요.
    불필요한 부사, 형용사, 대명사 표현을 자제해주세요.
    은유나 비유적인 표현이 아닌 직접적이고 정확한 표현을 사용하세요.

    컨텍스트: {context}

    질문: {input}

    답변:
    """
)

#! 체인 만들기 ---------------------------------------------------------------------------------
doc_chain = create_stuff_documents_chain(model, prompt)  # prompt | model
rag_chain = create_retrieval_chain(retriever, doc_chain)  # 검색 | doc_chain

#! --------------------------------- Gradio 챗봇 ---------------------------------------------
import gradio as gr


def answer_invoke(message, history):
    res = rag_chain.invoke({'input': message})
    return res['answer']


# Gradio 인터페이스 생성
demo = gr.ChatInterface(fn=answer_invoke, title='chatbot1')

demo.launch(share=True)

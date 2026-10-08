import os

import gradio as gr
from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import SecretStr

load_dotenv()

DB_PATH = './_db/FAISS19/'
BASE_URL = 'https://monogpt.kr/api/monorouter/v1/'
API_KEY = SecretStr(os.environ['MONOROUTER_API_KEY'])

#! 3. 임베딩 ------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=API_KEY,
    base_url=BASE_URL,
)
#! --------------------------------- FAISS ----------------------------------------------

#! LOAD ---------------------------------------------------------------------------------
db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index19',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

#! --------------------------------- 모델 연결 ---------------------------------------------
model = ChatOpenAI(
    model='gpt-6-luna',
    temperature=0,
    max_completion_tokens=1000,
    api_key=API_KEY,
    base_url=BASE_URL,
)

#! --------------------------------- Retrievers (검색기) ----------------------------------
prompt = ChatPromptTemplate.from_template(
    """
    # 역할
    당신은 "Attention Is All You Need" 논문을 해석하고 설명하는 전문가입니다.

    # 지침
    * 아래 [컨텍스트]에 있는 내용만 근거로 답변합니다. 컨텍스트에 없는 내용은 추측하거나 지어내지 않습니다.
    * 컨텍스트는 영어 논문에서 발췌한 것입니다. 한국어로 답변하되, 전문 용어는 영어 원문을 괄호로 병기합니다. (예: 다중 헤드 어텐션(Multi-Head Attention))
    * 결론을 첫 문장에 쓰고, 이어서 근거를 컨텍스트에서 인용해 설명합니다 (두괄식).
    * 근거를 인용할 때마다 문장 끝에 컨텍스트의 페이지 표시를 붙입니다. (예: (p.4))
    * 수치, 수식, 고유명사는 컨텍스트에 적힌 그대로 씁니다.
    * 질문의 일부만 컨텍스트에서 확인되면, 확인된 부분만 답하고 확인되지 않은 부분은 "주어진 정보로는 확인할 수 없습니다."라고 명시합니다.
    * 질문 전체에 대한 근거가 컨텍스트에 없으면 "주어진 정보로는 답변할 수 없습니다."라고만 답합니다.
    * 예시를 들 때는 컨  텍스트에 있는 수치나 문장을 풀어 쓰는 것으로 한정합니다.
    * 불필요한 부사, 형용사, 대명사와 은유, 비유를 쓰지 않고 직접적이고 정확한 표현을 사용합니다.

    # 컨텍스트
    {context}

    # 질문
    {input}

    # 답변
    """
)

# 컨텍스트에 본문만 넣으면 페이지를 인용할 수 없어서, 청크마다 [p.N] 을 붙인다
document_prompt = PromptTemplate.from_template('[p.{page_label}] {page_content}')

retriever = db.as_retriever(search_kwargs={'k': 4})

#! 체인 만들기 ---------------------------------------------------------------------------------
doc_chain = create_stuff_documents_chain(
    model, prompt, document_prompt=document_prompt
)  # prompt | model
rag_chain = create_retrieval_chain(retriever, doc_chain)  # 검색 | doc_chain


#! --------------------------------- Gradio 챗봇 ---------------------------------------------
def answer_invoke(message, history):
    res = rag_chain.invoke({'input': message})
    return res['answer']


demo = gr.ChatInterface(fn=answer_invoke, title='chatbot_attentionisallyouneed')

demo.launch(share=True)

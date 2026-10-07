import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
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
aaa = retriever.invoke(query)
print('------------------------------------------------------------')

#! --------------------------------- 모델 연결 ---------------------------------------------

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model='gpt-5-nano',
    temperature=0,
    max_completion_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

query_with_context = f"""
    {aaa[0].page_content}\n\n
    위 내용에 근거하여 다음 질문에 답변하세요 \n\n{query}
"""

res = model.invoke(query_with_context)
print('model의 답변: ', res.content)
print('------------------------------------------------------------')

# ? model의 답변:  삼성전자의 창업자는 이병철(이건희의 선조인 이병철, 영어로 Lee Byung-chul)입니다.
# ? 그는 1938년에 삼성그룹의 전신인 삼성상회를 설립했고,이후 삼성은 다각화된 기업으로 성장하여 삼성전자를 포함한 여러 계열사를 이끌었습니다.
# ? 임베딩된 데이터보다 model의 영향력이 강한 상태

# [학습 정리] rag01_basic 04 키가 제대로 들어왔는지 확인
# - os.environ['X'] 는 없으면 KeyError, os.getenv('X') 는 없으면 None → 확인용으로는 getenv 가 안전
# - 키 전체를 출력하지 않고 앞 8자 + 뒤 4자만 보여줘서 화면/로그 유출을 막는다

import os

key = os.getenv('OPENAI_API_KEY')

if key is None:
    print('OPENAI_API_KEY is None')
else:
    print('KEY LEN: ', len(key))
    print('KEY Check: ', key[:8] + '...' + key[-4:])

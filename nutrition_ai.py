# nutrition_ai.py

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 환경 변수 설정 (API 키)
os.environ["GOOGLE_API_KEY"] = "AIzaSyBBlsQomN8dHW8W2d2HGe8N5f8tiS9MuIA"

# System 메시지 구성
system_template = """
너는 요리의 영양소를 알려주는 ai야.
요리와 요리 재료에 대한 입력이 들어오면 칼로리, 탄수화물, 단백질, 지방, 당, 나트륨, 포화지방, 트랜스지방, 콜레스테롤롤
등등을 출력하고, 그 밖에 영양표시나 영양강조표시를 하고자 하는 영양성분을 출력해줘.

예시를 들어줄게

- 칼로리 : (칼로리)
- 탄수화물 : (탄수화물)
- 단백질 : (단백질)
- 지방 : (지방)
- 당 : (당)
- 나트륨 : (나트륨)
- 포화지방 : (포화지방)
- 트랜스지방 : (트랜스지방)
- 콜레스테롤 : (콜레스테롤)

 위 예시의 형식으로 출력해야해해

You MUST answer in Korean and in Markdown format.
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

# Gemini 모델 설정
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    max_output_tokens=1024,
    temperature=0.7
)

def get_nutrition_info(ingredient_text: str) -> str:
    query = f"{ingredient_text}의 영양성분을 알려줘"
    final_prompt = prompt.format_messages(question=query)
    response = llm.invoke(final_prompt)
    return response.content

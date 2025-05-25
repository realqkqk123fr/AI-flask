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
    query = f"""다음 재료와 요리에 대한 영양 정보를 정확하게 제공해주세요: {ingredient_text}
    
    ⚠️ 중요: 아래 형식을 정확히 지켜서 답변해주세요. 다른 설명 없이 오직 이 형식으로만 답변하세요.
    
    영양정보 형식:
    - 칼로리: [숫자]
    - 탄수화물: [숫자]
    - 단백질: [숫자] 
    - 지방: [숫자]
    - 당: [숫자]
    - 나트륨: [숫자]
    - 포화지방: [숫자]
    - 트랜스지방: [숫자]
    - 콜레스테롤: [숫자]
    
    주의사항:
    - 숫자만 입력하고 단위(g, mg, kcal)는 붙이지 마세요
    - 범위(450-600) 대신 평균값(525)을 사용하세요  
    - 괄호 안 설명을 추가하지 마세요
    - "약", "정도" 같은 부사를 사용하지 마세요
    - 미량인 경우 '0'으로 표시하세요
    - 각 라벨 뒤에는 반드시 콜론(:)을 붙이세요
    
    예시:
    - 칼로리: 520
    - 탄수화물: 65
    - 단백질: 18
    - 지방: 22
    - 당: 3
    - 나트륨: 450
    - 포화지방: 4
    - 트랜스지방: 0
    - 콜레스테롤: 380"""
    
    final_prompt = prompt.format_messages(question=query)
    response = llm.invoke(final_prompt)
    return response.content
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import RetrievalQAWithSourcesChain
from .vector_store import get_vector_store

def get_chain():
    system_template="""
    '너는 요리 레시피를 추천해주는 ai야.
    입력이 들어오면 그에 대해 최대한 잘 맞는 레시피를 단계별로 자세히게 추천해주었으면 좋겠어.
    모든 대답은 한국어(Korean)으로 대답해줘.
    아래 조건을 반드시 지켜줘:
    - 단계별로 자세히 설명
    - 설명 생략 금지 ("..." 쓰지 마)
    - 각 단계는 ### N단계 ### 형식으로
    - 모든 설명은 한국어로
    - 끝까지 출력하고 중단하지 마

    끝!!

    내가 예시를 들어줄게
    - name : (요리 이름)
    - description : (간단한 요리 설명)
    - ingredients : (요리 재료)
    - instructions : (단계별 요리 레시피)

    이런식으로 쭉 단계별로 끝까지 설명부탁해
    각 단계별로 자세히 설명 해줘
    ----------------
    {summaries}

    You MUST answer in Korean and in Markdown format:"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        max_output_tokens=4096,
        temperature=0.7,
        convert_system_message_to_human=True
    )

    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs=chain_type_kwargs
    )

    return chain

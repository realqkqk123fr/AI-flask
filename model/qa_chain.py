from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationalRetrievalChain
from .vector_store import get_vector_store
from langchain.memory import ConversationBufferWindowMemory

def get_chain():
    system_template="""
    너는 두가지 기능을 해야하는 ai 야

    첫번째로, 레시피를 소개해달라고 요청이 들어오면면

    그에 대해 최대한 잘 맞는 레시피를 단계별로 자세히게 추천해주었으면 좋겠어.
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

    두번째로 위 레시피에 관한 질문을 하면 최대한 친절하게 너가 가진 정보를 바탕으로 출력을 해주면 돼

    예시를 들면
    "이 레시피에서 간장을 빼고 대체할 수 있는 재료는 뭐야?"
    이런식으로 질문이 들어오면 너는 
    "간장 대신 레몬즙이나 소금으로 대체가능합니다!"

    - description : (대답)

    이런식으로 대답해주면 돼
    
    ----------------
    {context}


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

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",  # prompt에서 사용할 키
        k=3,  # 최근 3개 대화만 기억
        return_messages=True  # message 형식 유지
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs=chain_type_kwargs,
        output_key="answer"  # 이제 정상 작동
    )


    return chain

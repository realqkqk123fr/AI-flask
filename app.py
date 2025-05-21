from flask import Flask, request, jsonify, render_template
from cv_model import model
from PIL import Image, ImageDraw, ImageFont
import io, os, base64
import numpy as np
from config import *
from model.qa_chain import get_chain
from model.similarity_check import check_replace
from nutrition_ai import get_nutrition_info
import re
import uuid
import traceback

app = Flask(__name__)
qa_chain = get_chain()
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

id2name = model.names


#################################### 레시피 출력 LLM ###########################
"""
    입력 (예시)
    "question" : "대파, 마늘, 소안심 , 간단한 아침 식사로 먹기 좋을 요리로 부탁해"
    출력 (예시)
    {
     "description": "국간장 대신 진간장을 사용하여 더욱 깊고 풍부한 맛을 내는 돼지불고기 레시피입니다.",
     "ingredients": [ {"amount": "(불고기용) 600g", "name": "돼지고기"}, {"amount": "1개", "name": "양파"}, { ... }, { ... }],
     "instructions" : [ {"description": "돼지고기 준비: 불고기용 돼지고기를 키친타월로 꾹꾹 눌러 핏물을 제거합니다. 이렇게 하면 잡내를 줄이고 양념이 더 잘 배어들게 됩니다.", "step": 1},
                        {"description": "채소 준비: 양파는 얇게 채 썰고, 대파는 어슷하게 썰어줍니다. 당근을 사용한다면 얇게 채 썰어 준비합니다. 마늘과 생강은 곱게 다져줍니다.", "step": 2},
                         {...} ]
     "name": "진간장 돼지불고기",
     "user": null
    }
"""
# 세션별 대화 기록 저장을 위한 딕셔너리
conversation_history = {}

@app.route('/chat', methods=['POST'])
def chat():
    """
    일반 채팅 API: 사용자 메시지를 받아 AI 응답을 반환합니다.
    """
    
    try:
        # 인코딩 설정 확인
        print(f"요청 인코딩: {request.charset if hasattr(request, 'charset') else 'None'}")
        print(f"요청 Content-Type: {request.headers.get('Content-Type', 'None')}")

        # JSON 요청 처리
        if request.is_json:
            data = request.get_json()
            print(f"수신된 JSON 데이터: {data}")
        else:
            # 폼 데이터 처리 (이전 방식)
            data = {
                "message": request.form.get("message"),
                "username": request.form.get("username", "사용자"),
                "sessionId": request.form.get("sessionId", "")
            }
            print(f"수신된 폼 데이터: {data}")

        message = data.get("message")
        username = data.get("username", "사용자")
        session_id = data.get("sessionId", "")

        # 메시지 검증
        if not message:
            print("메시지가 비어 있음")
            return jsonify({
                "error": "메시지가 비어 있습니다",
                "message": "메시지를 입력해주세요."
            }), 400

        # 여기에 로깅 추가
        print(f"수신된 원본 메시지 (repr): {message!r}")
        print(f"수신된 원본 메시지 (type): {type(message)}")
        print(f"수신된 원본 메시지 (len): {len(message) if message else 0}")
        print(f"수신된 원본 메시지 (bytes): {message.encode('utf-8') if message else b''}")
        
        # 세션 ID가 없는 경우 생성 (클라이언트가 제공하지 않은 경우)
        if not session_id:
            session_id = str(uuid.uuid4())
            print(f"새 세션 ID 생성: {session_id}")
        
        print(f"사용자: {username}, 세션 ID: {session_id}")
        
        if not message:
            return jsonify({"error": "메시지가 비어 있습니다", "message": "메시지를 입력해주세요."}), 400
        
        try:
            # 세션 ID가 있으면 대화 기록 가져오기, 없으면 초기화
            if session_id not in conversation_history:
                print(f"새 대화 기록 초기화: {session_id}")
                # 대화 시작 시 시스템 메시지 설정
                conversation_history[session_id] = [
                    {
                        "role": "system", 
                        "content": """당신은 요리 AI 어시스턴트입니다.

중요한 지침:
1. 사용자가 레시피를 명시적으로 요청할 때만 상세한 레시피 형식으로 응답하세요.
2. 사용자가 단순히 인사하거나 일반적인 대화를 시도하면, 자연스럽게 대화하세요.
3. 사용자가 "안녕", "굿모닝", "있니?" 등과 같은 간단한 인사를 건넬 때는 절대로 레시피를 제공하지 마세요.
4. 이전 메시지와 동일한 응답을 반복하지 마세요. 대화가 진행됨에 따라 다양하게 응답하세요.
5. 먼저 사용자의 의도를 파악하고, 그에 맞게 응답하세요.
6. 대화 맥락을 유지하며 이전 대화를 참고하여 응답하세요.

레시피 형식(레시피 요청 시에만 사용):
- name: 레시피 이름
- description: 간단한 설명
- ingredients: 재료 목록
- instructions: 조리 방법

다시 한번 강조합니다: 사용자가 명시적으로 레시피를 요청하지 않았다면 위 형식을 사용하지 마세요."""
                    }
                ]
            
            # 이전 대화 내용 로깅
            print("이전 대화 내역:")
            for idx, item in enumerate(conversation_history[session_id]):
                print(f"{idx}. {item['role']}: {item['content'][:30]}...")
            
            # 사용자 메시지를 대화 기록에 추가
            conversation_history[session_id].append({"role": "user", "content": message})
            
            # 최근 4개 메시지만 포함하는 대화 맥락 구성 
            # (시스템 메시지 1개 + 사용자/어시스턴트 메시지 최대 3쌍)
            recent_messages = conversation_history[session_id][-7:] if len(conversation_history[session_id]) > 7 else conversation_history[session_id]
            
            # 여기에 대화 맥락 로깅 추가
            print("\n============= 대화 맥락 상세 =============")
            for idx, msg in enumerate(conversation_history[session_id]):
                role = msg["role"]
                content_preview = msg["content"][:50].replace('\n', ' ')
                content_preview += "..." if len(msg["content"]) > 50 else ""
                print(f"[{idx}] {role}: {content_preview}")
            print("=========================================\n")

            # LLM에 전송할 최종 프롬프트 구성
            messages = []

            # 시스템 메시지 추가
            for msg in recent_messages:
                if msg["role"] == "system":
                    messages.append({"role": "system", "content": msg["content"]})
                elif msg["role"] == "user":
                    messages.append({"role": "human", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    messages.append({"role": "ai", "content": msg["content"]})
            
            print(f"LLM에 전송되는 메시지 수: {len(messages)}")
            print(f"최신 사용자 메시지: {message}")
            
            try:
                # Google Gemini와 같은 LLM 모델에 직접 메시지로 전달
                from langchain_google_genai import ChatGoogleGenerativeAI
                from langchain.schema import HumanMessage, SystemMessage, AIMessage
                
                # 모델 초기화 (이미 qa_chain이 있으므로 여기서는 예시로만 제공)
                # llm = ChatGoogleGenerativeAI(model="gemini-pro")
                
                # 메시지 변환
                langchain_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        langchain_messages.append(SystemMessage(content=msg["content"]))
                    elif msg["role"] == "human":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "ai":
                        langchain_messages.append(AIMessage(content=msg["content"]))
                
                # 직접 LLM 호출 (실제로는 qa_chain을 사용하겠지만 예시로 제공)
                # response = llm.invoke(langchain_messages)
                
                # qa_chain을 사용하여 호출할 경우, 직접 질문 형식으로 변환
                combined_message = ""
                
                # 시스템 메시지 추가
                system_messages = [msg for msg in messages if msg["role"] == "system"]
                if system_messages:
                    combined_message += f"시스템 지시:\n{system_messages[0]['content']}\n\n"
                
                # 대화 내역 추가 (시스템 메시지 제외)
                combined_message += "대화 내역:\n"
                for msg in messages:
                    if msg["role"] != "system":
                        role_name = "사용자" if msg["role"] == "human" else "AI"
                        combined_message += f"{role_name}: {msg['content']}\n"
                
                # 프롬프트 마무리
                combined_message += "\n이제 사용자의 마지막 메시지에 대해 응답하세요."
                
                print(f"최종 프롬프트:\n{combined_message[:200]}...")
                
                # qa_chain 호출
                result = qa_chain.invoke({"question": combined_message})
                response_text = result["answer"]
                
            except Exception as chain_error:
                print(f"LLM 호출 중 오류 발생: {str(chain_error)}")
                # 원래 방식으로 폴백
                result = qa_chain.invoke({"question": message})
                response_text = result["answer"]
            
            print(f"LLM 응답: {response_text[:100]}...")
            
            # 여기에 응답 로깅 추가
            print(f"\n========== LLM 응답 원본 ==========")
            print(f"{response_text}")
            print(f"===================================\n")

            # AI 응답을 대화 기록에 추가
            conversation_history[session_id].append({"role": "assistant", "content": response_text})
            
            # 대화 기록 길이 제한 (시스템 메시지 유지, 나머지는 최근 10개만)
            if len(conversation_history[session_id]) > 11:  # 시스템 메시지 + 10개 대화
                system_msg = conversation_history[session_id][0]  # 시스템 메시지 보존
                conversation_history[session_id] = [system_msg] + conversation_history[session_id][-10:]
            
            # 현재 대화 상태 로깅
            print(f"대화 내역 길이(업데이트 후): {len(conversation_history[session_id])}")
            
            # 응답 처리 및 정리 (마크다운 및 description 형식 처리)
            cleaned_message = response_text
            
            # 여기에 정리 과정 로깅 추가
            print(f"\n========== 메시지 정리 과정 ==========")
            print(f"원본 메시지: {cleaned_message[:100]}...")

            # Markdown 헤더(##) 제거
            cleaned_message = re.sub(r'^##\s+', '', cleaned_message)
            print(f"헤더 제거 후: {cleaned_message[:100]}...")

            # description: 형식 추출
            desc_match = re.search(r'- description\s*:\s*(.+?)(?=$|\n\n)', cleaned_message, re.DOTALL)
            if desc_match:
                # description 내용만 추출
                cleaned_message = desc_match.group(1).strip()
                print(f"description 추출: {cleaned_message[:100]}...")
            else:
                # description 패턴이 없으면 전체 텍스트 사용 (불필요한 마크다운 제거)
                print("description 패턴 없음, 마크다운 제거")
                cleaned_message = re.sub(r'- \w+\s*:\s*', '', cleaned_message)
                cleaned_message = re.sub(r'^\s*\*\s*', '', cleaned_message, flags=re.MULTILINE)
                cleaned_message = re.sub(r'###\s*\d+단계\s*###', '', cleaned_message)
                print(f"마크다운 제거 후: {cleaned_message[:100]}...")

            # 여러 줄 공백 제거
            cleaned_message = re.sub(r'\n{3,}', '\n\n', cleaned_message)
            print(f"최종 정리된 메시지: {cleaned_message[:100]}...")
            print(f"=====================================\n")

            print(f"정리된 메시지: {cleaned_message[:100]}...")
            
            # 최종 응답 로깅
            final_response = {
                "message": cleaned_message,
                "username": "AI 요리사",
                "sessionId": session_id
            }
            print(f"\n========== 최종 응답 JSON ==========")
            import json
            print(json.dumps(final_response, ensure_ascii=False, indent=2))
            print(f"===================================\n")

            # 응답 반환
            return jsonify(final_response)
                    
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            traceback.print_exc()  # 상세 스택 트레이스 출력
            return jsonify({
                "error": "응답 생성 중 오류가 발생했습니다",
                "message": "죄송합니다, 요청을 처리하는 중에 문제가 발생했습니다. 다시 시도해주세요.",
                "username": "AI 요리사"
            }), 500
            
    except Exception as e:
        print(f"요청 파싱 오류: {str(e)}")
        traceback.print_exc()  # 상세 스택 트레이스 출력
        return jsonify({
            "error": "요청 처리 중 오류가 발생했습니다",
            "message": f"요청 형식이 올바르지 않습니다: {str(e)}",
            "username": "AI 요리사"
        }), 400
                    
# 레시피 이름 추출
def extract_name(text):
    match = re.search(r"- name\s*:\s*(.+)", text)
    return match.group(1).strip() if match else "이름 없음"

# 설명 추출
def extract_description(text):
    match = re.search(r"- description\s*:\s*(.+)", text)
    return match.group(1).strip() if match else "설명 없음"

# 재료 리스트 추출
def extract_ingredients(text):
    ingredients = []
    match = re.search(r"- ingredients\s*:\s*((?:\n\s*\*.+)+)", text)
    if match:
        raw = match.group(1).strip().split('\n')
        for line in raw:
            # 줄 시작 부분의 별표와 공백 제거
            item = re.sub(r"^\s*\*\s*", "", line).strip()
            if item:
                # 콜론으로 구분된 이름과 수량 (기존 로직)
                parts = item.split(":", 1)
                if len(parts) == 2:
                    name, amount = parts[0].strip(), parts[1].strip()
                else:
                    # 수량 정보가 없는 경우: 수량 추정 로직 추가
                    # 형식이 "재료명 수량" 형태인지 확인 (예: 양파 1개, 소금 약간)
                    name_amount_match = re.match(r"^(.+?)(\d+[^\d\s]+|\d+\s*[^\d\s]+|약간|소량|적당량)$", item)
                    if name_amount_match:
                        name, amount = name_amount_match.groups()
                        name = name.strip()
                        amount = amount.strip()
                    else:
                        name, amount = item, "적당량"  # 기본값 설정
                
                # 이름에 별표가 포함된 경우 제거
                name = name.replace("*", "").strip()
                
                # 수량이 비어있으면 "적당량"으로 설정
                if not amount or amount.strip() == "":
                    amount = "적당량"
                
                ingredients.append({"name": name, "amount": amount})
    return ingredients

# 조리 단계 추출
def extract_instructions(text):
    instructions = []
    matches = re.findall(r"###\s*(\d+)단계\s*###\n(.+?)(?=\n###|\Z)", text, re.DOTALL)
    
    for step_num, step_content in matches:
         step_text = step_content.strip()
        
         # 조리 시간 추출에 새로운 함수 활용
         cooking_time_mins, cooking_time_seconds = extract_cooking_time(step_text)
        
         instructions.append({
             "step": int(step_num),
             "text": step_text,
             "cookingTime": cooking_time_mins,  # 분 단위로 저장
             "cookingTimeSeconds": cooking_time_seconds  # 초 단위로 저장
         })
    return instructions

def extract_cooking_time(text):
    """
    텍스트에서 조리 시간을 추출하거나 예상하는 함수
    
    Args:
        text (str): 조리 단계 텍스트
        
    Returns:
        tuple: (분 단위 시간, 초 단위 시간)
    """
    # 기본값 설정
    cooking_time_mins = 5  # 기본 5분
    cooking_time_seconds = 300  # 기본 300초
    
    # 텍스트에서 "N분" 또는 "N분 M초" 패턴 추출 시도
    time_match = re.search(r'(\d+)\s*분(?:\s*(\d+)\s*초)?', text)
    if time_match:
        minutes = int(time_match.group(1))
        seconds = int(time_match.group(2)) if time_match.group(2) and time_match.group(2).strip() else 0
        cooking_time_mins = minutes
        cooking_time_seconds = minutes * 60 + seconds
    else:
        # 패턴이 없는 경우 텍스트 내용에 따라 시간 추정
        word_count = len(text.split())
        
        if "볶" in text or "굽" in text:
            # 볶거나 굽는 작업은 3-10분 소요, 단어 수에 따라 조정
            cooking_time_seconds = min(10 * 60, max(3 * 60, word_count * 20))  # 3-10분
            cooking_time_mins = cooking_time_seconds // 60
        elif "끓" in text or "삶" in text:
            # 끓이거나 삶는 작업은 5-15분 소요, 단어 수에 따라 조정
            cooking_time_seconds = min(15 * 60, max(5 * 60, word_count * 30))  # 5-15분
            cooking_time_mins = cooking_time_seconds // 60
        elif "썰" in text or "다듬" in text or "준비" in text:
            # 썰거나 준비하는 작업은 1-5분 소요, 단어 수에 따라 조정
            cooking_time_seconds = min(5 * 60, max(1 * 60, word_count * 10))  # 1-5분
            cooking_time_mins = cooking_time_seconds // 60
        elif "식히" in text or "숙성" in text:
            # 식히거나 숙성하는 작업은 10분 정도 소요
            cooking_time_seconds = 10 * 60  # 10분
            cooking_time_mins = 10
        # 기본값은 이미 설정됨 (5분, 300초)
    
    return cooking_time_mins, cooking_time_seconds

# 기존 extract_instructions 함수 이후에 추가 처리 단계 구현
def process_instruction_steps(instructions_raw):
    processed_instructions = []
    
    for instruction in instructions_raw:
        step_num = instruction.get("step", 0)
        text = instruction.get("text", "")
        
        # 내부에 있는 ### N단계 ### 패턴 확인
        inner_steps = re.findall(r"###\s*(\d+)단계\s*###\n(.+?)(?=\n###|\Z)", text, re.DOTALL)
        
        if inner_steps:
            # 내부 단계 패턴이 있는 경우
            for inner_step, inner_text in inner_steps:
                inner_text = inner_text.strip()
                
                # 조리 시간 추출
                cooking_time_mins, cooking_time_seconds = extract_cooking_time(inner_text)
                
                processed_instructions.append({
                    "instruction": inner_text,
                    "cookingTime": cooking_time_mins,
                    "cookingTimeSeconds": cooking_time_seconds,
                    "stepNumber": int(inner_step)
                })
        else:
            # 내부 단계 패턴이 없는 경우, 길이 제한 적용
            if len(text) > 200:
                # 문장 단위로 분할
                sentences = re.split(r'(?<=[.!?])\s+', text)
                current_chunk = ""
                chunk_counter = 0
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= 200:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        # 첫 번째 청크만 조리 시간 추출
                        if chunk_counter == 0:
                            cooking_time_mins, cooking_time_seconds = extract_cooking_time(current_chunk)
                        else:
                            cooking_time_mins, cooking_time_seconds = 0, 0  # 첫 번째가 아닌 청크는 시간 0
                        
                        processed_instructions.append({
                            "instruction": current_chunk.strip(),
                            "cookingTime": cooking_time_mins,
                            "cookingTimeSeconds": cooking_time_seconds,
                            "stepNumber": step_num * 100 + chunk_counter
                        })
                        chunk_counter += 1
                        current_chunk = sentence
                
                # 마지막 청크 저장
                if current_chunk:
                    if chunk_counter == 0:
                        cooking_time_mins, cooking_time_seconds = extract_cooking_time(current_chunk)
                    else:
                        cooking_time_mins, cooking_time_seconds = 0, 0
                    
                    processed_instructions.append({
                        "instruction": current_chunk.strip(),
                        "cookingTime": cooking_time_mins,
                        "cookingTimeSeconds": cooking_time_seconds,
                        "stepNumber": step_num * 100 + chunk_counter
                    })
            else:
                # 짧은 텍스트는 그대로 저장
                cooking_time_mins, cooking_time_seconds = extract_cooking_time(text)
                
                processed_instructions.append({
                    "instruction": text,
                    "cookingTime": cooking_time_mins,
                    "cookingTimeSeconds": cooking_time_seconds,
                    "stepNumber": step_num
                })
    
    return processed_instructions

####################### 대체재료 기반 LLM ###############################

"""
    입력 (예시)
    {
    "ori": "국간장",
    "sub": "진간장",
    "recipe": "간장돼지불고기"
    }
    출력 (예시)
    {
     "description": "국간장 대신 진간장을 사용하여 더욱 깊고 풍부한 맛을 내는 돼지불고기 레시피입니다.",
     "ingredients": [ {"amount": "(불고기용) 600g", "name": "돼지고기"}, {"amount": "1개", "name": "양파"}, { ... }, { ... }],
     "instructions" : [ {"description": "돼지고기 준비: 불고기용 돼지고기를 키친타월로 꾹꾹 눌러 핏물을 제거합니다. 이렇게 하면 잡내를 줄이고 양념이 더 잘 배어들게 됩니다.", "step": 1},
                        {"description": "채소 준비: 양파는 얇게 채 썰고, 대파는 어슷하게 썰어줍니다. 당근을 사용한다면 얇게 채 썰어 준비합니다. 마늘과 생강은 곱게 다져줍니다.", "step": 2},
                         {...} ]
     "name": "진간장 돼지불고기",
     "user": null
    }
"""

@app.route('/generate_recipe_or_reject', methods=['POST'])
def generate_recipe_or_reject():
    data = request.get_json()
    ori = data.get("ori")
    sub = data.get("sub")
    recipe = data.get("recipe")

    if not all([ori, sub, recipe]):
        return jsonify({"error": "요청 필드가 부족합니다."}), 400

    try:
        similarity_score = check_replace(ori, sub)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if similarity_score < 0.6:
        return jsonify({
            "name": recipe,
            "description": f"{ori}를 {sub}로 대체하는 것은 적절하지 않아 레시피를 생성할 수 없습니다.",
            "ingredients": [],
            "instructions": [],
            "user": None
        })

    # LLM 질의 및 응답
    query = f"{ori}를 {sub}로 교체한 {recipe}의 레시피를 알려줘"
    result = qa_chain.invoke({"question": query})
    raw = result["answer"]

    # 🔍 파싱 시작
    try:
        name = re.search(r'- name *: *(.*)', raw).group(1).strip()
        description = re.search(r'- description *: *(.*)', raw).group(1).strip()

        # ingredients 파싱
        ingredients_raw = re.findall(r'\* *(.*)', raw)
        ingredients = []
        for item in ingredients_raw:
            parts = item.split(' ', 1)
            if len(parts) == 2:
                ingredients.append({"name": parts[0], "amount": parts[1]})
            else:
                ingredients.append({"name": parts[0], "amount": ""})

        # instructions 파싱
        instructions_raw = re.findall(r'### *\d+단계 *###\n(.+?)(?=\n###|\Z)', raw, re.DOTALL)
        instructions = [
            {"step": idx + 1, "description": step.strip()}
            for idx, step in enumerate(instructions_raw)
        ]

        # 응답 형식 수정
        instructions_processed = []
        for idx, step in enumerate(instructions_raw):
            step_text = step.strip()
    
            # 조리 시간 추출에 새로운 함수 활용
            cooking_time_mins, cooking_time_seconds = extract_cooking_time(step_text)
            
            instructions_processed.append({
                "instruction": step_text,
                "cookingTime": cooking_time_mins,
                "cookingTimeSeconds": cooking_time_seconds,
                "stepNumber": idx + 1
            })
        
        response_json = {
            "name": name,
            "description": description,
            "ingredients": ingredients,
            "instructions": instructions_processed,  # 변환된 형식
            "user": None
        }

        return jsonify(response_json)

    except Exception as e:
        return jsonify({"error": f"레시피 파싱 중 오류 발생: {str(e)}"}), 500


######################## 영양소 출력 LLM #####################################
"""
    입력 (예시)
    {
    "ingredients" : "소안심200g, 대파 1대, 마늘 5쪽, 간장 1큰술, 굴소스 1/2큰술, 참기름 1/2큰술, 후추 약간, 식용유 적당량, 소고기 대파 마늘볶음"
    }
    출력 (예시)
    {
    "calories": 500.0,
    "carbohydrate": 12.5,
    "cholesterol": 0.0,
    "fat": 30.0,
    "protein": 45.0,
    "saturatedFat": 0.0,
    "sodium": 600.0,
    "sugar": 6.5,
    "transFat": 0.0
}
"""

@app.route("/nutrition", methods=["POST"])
def nutrition():
    try:
        data = request.get_json()
        ingredients = data.get("ingredients")

        if not ingredients:
            return jsonify({"error": "No 'ingredients' field in request"}), 400

        response_text = get_nutrition_info(ingredients)
        print("🧠 모델 응답:\n", response_text)
        if not response_text:
            return jsonify({"error": "모델 응답이 비었습니다."}), 500

        result = extract_nutrition(response_text)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def extract_nutrition(text):
    def extract_value(pattern, default=0.0):
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            # 디버깅을 위해 추출된 값 로깅
            print(f"추출된 값 ({pattern}): {value}")
            if '-' in value or '~' in value:
                # 범위 값 처리 개선
                parts = re.split(r"[-~]", value)
                # 숫자만 추출 (단위 제거)
                nums = []
                for p in parts:
                    # 숫자 부분만 추출
                    num_str = re.sub(r"[^0-9.]", "", p.strip())
                    if num_str:
                        try:
                            nums.append(float(num_str))
                        except ValueError:
                            print(f"숫자 변환 실패: {num_str}")
                
                # 결과 로깅 및 계산
                print(f"추출된 숫자: {nums}")
                return sum(nums) / len(nums) if nums else default
            
            # 단일 값인 경우 - 단위 제거
            value = re.sub(r"[^0-9.]", "", value)
            return float(value) if value else default
        
        # 일치하는 패턴이 없는 경우
        print(f"패턴 미일치: {pattern}")
        return default

    # 기존 패턴 개선
    result = {
        "calories": extract_value(r"칼로리:?\s*약?\s*([\d\-~]+)\s*kcal"),      # kcal
        "carbohydrate": extract_value(r"탄수화물:?\s*약?\s*([\d\-~]+)\s*g"),   # g  
        "protein": extract_value(r"단백질:?\s*약?\s*([\d\-~]+)\s*g"),         # g
        "fat": extract_value(r"지방:?\s*약?\s*([\d\-~]+)\s*g"),              # g
        "sugar": extract_value(r"당:?\s*약?\s*([\d\-~]+)\s*g"),              # g
        "sodium": extract_value(r"나트륨:?\s*약?\s*([\d\-~]+)\s*mg"),         # mg
        "saturatedFat": extract_value(r"포화지방:?\s*약?\s*([\d\-~]+)\s*g"),   # g
        "transFat": extract_value(r"트랜스지방:?\s*약?\s*([\d.]+|미량)\s*g?"),  # g
        "cholesterol": extract_value(r"콜레스테롤:?\s*약?\s*([\d\-~]+)\s*mg")  # mg
    }
    
    # 디버깅을 위한 결과 로깅
    print(f"추출된 영양 정보: {result}")
    
    return result

## 새로운 통합 엔드포인트 - 이미지 분석 + 레시피 생성
@app.route("/analyze_and_generate_recipe", methods=["POST"])
def analyze_and_generate_recipe():
    try:
        print("=== 요청 시작 ===")
        print(f"Instructions: {request.form.get('instructions', '')}")
        print(f"Instructions 타입: {type(request.form.get('instructions', ''))}")
        print(f"Instructions 길이: {len(request.form.get('instructions', ''))}")
        print(f"Instructions 바이트: {request.form.get('instructions', '').encode('utf-8')}")
        print(f"Username: {request.form.get('username', '')}")

        # 이미지 파일 확인
        if 'image' not in request.files:
            return jsonify({"error": "이미지 파일이 제공되지 않았습니다."}), 400
        
        # 지시사항 확인
        instructions = request.form.get('instructions', '')
        username = request.form.get('username', '사용자')
        
        # 세션 ID (필요한 경우)
        session_id = request.form.get('sessionId', '')
        
        # 이미지 저장
        image_file = request.files['image']

        # 파일 이름 안전하게 처리
        import os
        import uuid
        from werkzeug.utils import secure_filename

        # 원본 파일명에서 확장자 추출 (소문자로 변환)
        original_filename = image_file.filename
        print(f"원본 파일명: {original_filename}")
        file_ext = os.path.splitext(original_filename)[1].lower()  # 소문자로 변환

        # 확장자가 없거나 비어있는 경우 기본 확장자 추가
        if not file_ext:
            file_ext = '.jpg'  # 기본 이미지 확장자
            print(f"확장자가 없어 기본값 적용: {file_ext}")
        else:
            print(f"추출된 확장자: {file_ext}")

        # 안전한 파일명 생성
        safe_filename = f"{uuid.uuid4().hex}{file_ext}"
        print(f"생성된 안전한 파일명: {safe_filename}")

        # 저장 경로 생성
        path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        print(f"이미지 저장 경로: {path}")

        # 파일 저장
        image_file.save(path)

        # 파일이 실제로 저장되었는지 확인
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            print(f"이미지 저장 성공: {path} (크기: {file_size} 바이트)")
        else:
            print(f"이미지 저장 실패: {path}")
            return jsonify({"error": "이미지 파일 저장에 실패했습니다."}), 500

        # 선택적: 이미지 유효성 검사 (PIL이 이미 임포트되어 있다고 가정)
        try:
            img_check = Image.open(path)
            img_check.verify()  # 빠른 검증
            print(f"이미지 검증 성공: 형식={img_check.format}")
        except Exception as e:
            print(f"이미지 검증 실패: {str(e)}")
            # 검증 실패해도 계속 진행 (모델이 처리할 수도 있으므로)
        
        # 1. YOLO 모델을 사용한 이미지 분석
        results = model.predict(source=path, imgsz=640, conf=0.25)
        box_data = results[0].cpu().boxes
        
        # 감지된 객체 추출
        xyxy = box_data.xyxy.cpu().numpy()   # [[x1,y1,x2,y2], ...]
        confs = box_data.conf.cpu().numpy()  # [conf, ...]
        cls_ids = box_data.cls.cpu().numpy() # [class_id, ...]
        
        detected_objects = []
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, cls_ids):
            detected_objects.append({
                'class_id': int(cls),
                'class_name': id2name[int(cls)],
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
        
        # 2. 감지된 객체의 이름만 추출 (신뢰도 0.25 이상으로 낮춤)
        detected_ingredients = [obj['class_name'] for obj in detected_objects 
                               if obj['confidence'] > 0.25]  # 신뢰도 임계값 낮춤
        
        # 3. 여기가 수정할 부분입니다! 사용자 지시사항과 감지된 객체를 결합하여 LLM 쿼리 생성
        combined_query = ""
        if detected_ingredients:
            ingredients_str = ", ".join(detected_ingredients)
            
            # 프롬프트 개선 - 명확한 포맷 지정
            combined_query = f"""
            {instructions}

            이미지에서 감지된 재료: {ingredients_str}

            다음 형식으로 레시피를 제공해주세요:
            - name: 레시피 이름
            - description: 간단한 설명
            - ingredients:
            * 재료1: 수량 (예: 양파 1개, 간장: 2큰술, 고춧가루: 1큰술)
            * 재료2: 수량
            * 재료3: 수량
            - instructions:
            ### 1단계 ###
            첫 번째 조리 단계 설명. 각 단계에 예상 소요 시간을 꼭 명시하세요. 예: "양파를 다진 후 3분간 볶아주세요."
    
            ### 2단계 ###
            두 번째 조리 단계 설명. 소요 시간 명시. 예: "물 500ml를 붓고 10분간 끓여주세요."
    
            ### 3단계 ###
            세 번째 조리 단계 설명. 소요 시간 명시. 예: "고기를 넣고 5분간 더 끓인 후 간을 맞춰주세요."
            
            ...

            중요: 각 단계에 예상 소요 시간을 명확히 표시하세요 (예: "3분간 볶는다", "10분간 끓인다"). 이는 사용자의 타이머 설정에 사용됩니다. 각 단계는 최대 100자로 간결하게 작성하세요. 한 단계 내에 다른 단계(### N단계 ###)를 포함하지 마세요.
            """
        else:
            combined_query = f"""
            {instructions}

            다음 형식으로 레시피를 제공해주세요:
            - name: 레시피 이름
            - description: 간단한 설명
            - ingredients:
            * 재료1: 수량
            * 재료2: 수량
            * 재료3: 수량
             - instructions:
            ### 1단계 ###
            짧고 명확한 첫 번째 조리 단계 설명 (한 단계당 최대 100자)
            
            ### 2단계 ###
            짧고 명확한 두 번째 조리 단계 설명 (한 단계당 최대 100자)
            
            ### 3단계 ###
            짧고 명확한 세 번째 조리 단계 설명 (한 단계당 최대 100자)
            
            ...

            중요: 각 단계는 최대 100자로 간결하게 작성하세요. 한 단계 내에 다른 단계(### N단계 ###)를 포함하지 마세요.
            """
        
        # 로깅 추가
        print(f"LLM에 보내는 쿼리: {combined_query}")
        
        # 인코딩 문제를 피하기 위해 UTF-8로 명시적 변환
        combined_query_utf8 = combined_query.encode('utf-8').decode('utf-8')

        # 4. LLM을 사용하여 레시피 생성
        result = qa_chain.invoke({"question": combined_query_utf8})
        raw_text = result["answer"]
        
        # 로깅 추가
        print(f"LLM 응답: {raw_text}")
        
        # 5. 결과 파싱
        try:
            # 레시피 이름 추출
            name = extract_name(raw_text)
            # 설명 추출
            description = extract_description(raw_text)
            # 재료 추출
            ingredients = extract_ingredients(raw_text)
            # 지시사항 추출 - 직접 패턴 찾기
            processed_instructions = []
            step_pattern = r"###\s*(\d+)단계\s*###\s*(.*?)(?=###\s*\d+단계\s*###|\Z)"
            steps = re.findall(step_pattern, raw_text, re.DOTALL)
            
            print(f"단계 추출 결과: {len(steps)} 단계 발견")
            
            for step_num, step_content in steps:
                step_text = step_content.strip()
                if step_text:
                    # 조리 시간 추출
                    cooking_time_mins, cooking_time_seconds = extract_cooking_time(step_text)
                    print(f"단계 {step_num}에서 조리 시간: {cooking_time_mins}분 ({cooking_time_seconds}초)")
                    
                    # 이 부분이 중요: stepNumber를 일반적인 숫자로 설정
                    processed_instructions.append({
                        "instruction": step_text,
                        "cookingTime": cooking_time_mins,
                        "cookingTimeSeconds": cooking_time_seconds,
                        "stepNumber": int(step_num)
                    })
                    print(f"단계 {step_num}: {step_text[:50]}...")

            # 재료나 조리 단계가 비어있는 경우 기본값 제공
            if not ingredients:
                ingredients = [{"name": ing, "amount": "적당량"} for ing in detected_ingredients]
            
            if not instructions:
                instructions = [{"step": 1, "text": "감지된 재료로 요리하는 레시피를 생성 중입니다. 다시 시도해 주세요."}]
            
            # 6. 이미지 처리 (Base64로 인코딩)
            # 원본 이미지에 감지된 객체 표시
            img = Image.open(path).convert('RGB')
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", size=16)
            except:
                font = ImageFont.load_default()

            # 객체 바운딩 박스 그리기
            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, cls_ids):
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                label = f"{id2name[int(cls)]}:{conf:.2f}"
                
                x_min, y_min, x_max, y_max = font.getbbox(label)
                text_width = x_max - x_min
                text_height = y_max - y_min

                text_size = [text_width, text_height]
                draw.rectangle([x1, y1 - text_size[1], x1 + text_size[0], y1], fill="green")
                draw.text((x1, y1 - text_size[1]), label, fill="white", font=font)

            # 이미지를 Base64로 인코딩
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            img_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            
            # 7. 결과 구성
            print("=== 응답 생성 ===")
            print(f"Name: {name}")
            print(f"Description: {description}")
            print(f"Ingredients count: {len(ingredients)}")
            print(f"Instructions count: {len(processed_instructions)}")

            # 총 조리 시간 계산
            total_cooking_time_seconds = sum(instruction.get("cookingTimeSeconds", 0) for instruction in processed_instructions)
            total_cooking_time_mins = total_cooking_time_seconds // 60

            # 응답 JSON에 추가
            response_json = {
                "name": name,
                "description": description,
                "ingredients": ingredients,
                "instructions": processed_instructions,
                "imageUrl": f"data:image/jpeg;base64,{img_b64}",
                "user": {
                    "username": username
                },
                "sessionId": session_id,
                "totalCookingTime": total_cooking_time_mins,  # 총 조리 시간(분) 추가
                "totalCookingTimeSeconds": total_cooking_time_seconds  # 총 조리 시간(초) 추가
            }
            
            print("응답 구조:")
            print(f"이름: {name}")
            print(f"설명: {description}")
            print(f"재료 수: {len(ingredients)}")
            print(f"지시사항 수: {len(processed_instructions)}")
            
            return jsonify(response_json)
            
        except Exception as e:
            return jsonify({"error": f"레시피 생성 결과 파싱 오류: {str(e)}"}), 500
        
    except Exception as e:
        import traceback
        print("=== 오류 발생 ===")
        traceback.print_exc() # 자세한 오류 출력
        return jsonify({"error": f"처리 중 오류 발생: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
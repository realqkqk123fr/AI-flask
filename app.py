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

# 🔧 조리 시간 추출 함수도 개선
def extract_cooking_time(text):
    """
    텍스트에서 조리 시간을 추출하거나 예상하는 함수 (수정된 버전)
    """
    # 기본값 설정
    cooking_time_mins = 5  # 기본 5분
    cooking_time_seconds = 300  # 기본 300초 (5분)
    
    # 명시적 시간 패턴 추출 - 더 많은 패턴 지원
    time_patterns = [
        r'(\d+)\s*분(?:\s*(\d+)\s*초)?',  # N분 M초
        r'약\s*(\d+)\s*분(?:\s*(\d+)\s*초)?',  # 약 N분 M초
        r'(\d+)분간',                  # N분간
        r'약\s*(\d+)\s*분',               # 약 N분
        r'(\d+)-(\d+)\s*분',              # N-M분
        r'(\d+)\s*초',                     # N초
        r'(\d+)초간',                     # N초간
        r'약\s*(\d+)\s*초'                # 약 N초
    ]
    
    for pattern in time_patterns:
        time_match = re.search(pattern, text)
        if time_match:
            groups = time_match.groups()
            
            # "N초" 패턴인 경우
            if '초' in pattern and '분' not in pattern:
                seconds = int(groups[0])
                cooking_time_seconds = seconds
                cooking_time_mins = seconds // 60  # 초를 분으로 변환 (나눗셈의 몫)
                
                print(f"초 단위 시간 추출: {seconds}초 = {cooking_time_mins}분 {seconds % 60}초")
                return cooking_time_mins, cooking_time_seconds
            
            # "N분 M초" 패턴인 경우
            elif len(groups) >= 2 and groups[1] is not None:
                minutes = int(groups[0])
                seconds = int(groups[1])
                cooking_time_mins = minutes
                cooking_time_seconds = minutes * 60 + seconds
                
                print(f"분/초 단위 시간 추출: {minutes}분 {seconds}초 = {cooking_time_seconds}초")
                return cooking_time_mins, cooking_time_seconds
            
            # "N분" 패턴인 경우
            else:
                minutes = int(groups[0])
                cooking_time_mins = minutes
                cooking_time_seconds = minutes * 60
                
                print(f"분 단위 시간 추출: {minutes}분 = {cooking_time_seconds}초")
                return cooking_time_mins, cooking_time_seconds
    
    # 패턴이 없는 경우 텍스트 내용에 따라 시간 추정 (개선된 로직)
    word_count = len(text.split())
    
    if any(keyword in text for keyword in ["볶", "굽", "지글지글"]):
        cooking_time_seconds = min(10 * 60, max(2 * 60, word_count * 15))  # 2-10분
    elif any(keyword in text for keyword in ["끓", "삶", "우려"]):
        cooking_time_seconds = min(15 * 60, max(5 * 60, word_count * 25))  # 5-15분
    elif any(keyword in text for keyword in ["썰", "다듬", "준비", "씻"]):
        cooking_time_seconds = min(5 * 60, max(1 * 60, word_count * 8))   # 1-5분
    elif any(keyword in text for keyword in ["식히", "숙성", "재우"]):
        cooking_time_seconds = 10 * 60  # 10분
    elif any(keyword in text for keyword in ["섞", "젓", "휘젓"]):
        cooking_time_seconds = min(3 * 60, max(30, word_count * 5))       # 30초-3분
    else:
        # 기본 추정: 텍스트 길이에 비례
        cooking_time_seconds = max(60, min(8 * 60, word_count * 10))       # 1-8분
    
    cooking_time_mins = cooking_time_seconds // 60  # 올바른 분 단위 계산 (소수점 버림)
    
    print(f"시간 추정: {cooking_time_mins}분 ({cooking_time_seconds}초) - 키워드 기반")
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
    try:
        # 요청 데이터 검증
        if not request.is_json:
            print("JSON 형식이 아닌 요청 수신")
            return jsonify({
                "error": "JSON 형식의 요청이 필요합니다.",
                "substituteFailure": True
            }), 400

        data = request.get_json()
        
        # 필수 필드 검증
        ori = data.get("ori", "").strip() if data.get("ori") else ""
        sub = data.get("sub", "").strip() if data.get("sub") else ""
        recipe = data.get("recipe", "").strip() if data.get("recipe") else ""
        
        # 기존 레시피 데이터
        original_recipe_data = data.get("originalRecipe", {})
        original_ingredients = original_recipe_data.get("ingredients", [])
        original_instructions = original_recipe_data.get("instructions", [])

        # 입력값 로깅
        print(f"대체 재료 요청 - 원재료: '{ori}', 대체재료: '{sub}', 레시피: '{recipe}'")
        print(f"기존 레시피 재료 수: {len(original_ingredients)}")

        # 필수 필드 검증
        if not ori or not sub or not recipe:
            error_msg = "원재료, 대체재료, 레시피명을 모두 입력해주세요."
            print(f"필수 필드 누락: {error_msg}")
            return jsonify({
                "error": error_msg,
                "substituteFailure": True
            }), 400

        # 같은 재료인지 확인
        if ori.lower() == sub.lower():
            error_msg = f"같은 재료({ori})로는 대체할 수 없습니다."
            print(f"동일 재료 대체 시도: {error_msg}")
            return jsonify({
                "error": error_msg,
                "substituteFailure": True
            }), 200

        # LLM을 사용한 대체 가능성 평가 및 레시피 생성
        is_substitute_possible, updated_recipe = evaluate_substitute_with_llm(ori, sub, recipe, original_recipe_data)
        
        if not is_substitute_possible:
            # 대체 불가능으로 판단된 경우
            error_msg = f"{ori}를 {sub}로 대체하는 것은 적절하지 않습니다. {updated_recipe.get('reason', '재료의 특성이 너무 달라 맛과 식감에 큰 영향을 줄 수 있습니다.')}"
            print(f"LLM 대체 불가능 판단: {error_msg}")
            
            # 응답에 명시적으로 substituteFailure 플래그 설정
            return jsonify({
                "name": recipe,
                "description": error_msg,
                "ingredients": [],
                "instructions": [],
                "substituteFailure": True,  # 명시적 실패 플래그
                "reason": updated_recipe.get("reason", "")
            }), 200

        # 대체 가능한 경우 응답 구성
        response_json = {
            "name": updated_recipe.get("name", f"{sub}를 사용한 {recipe}"),
            "description": updated_recipe.get("description", f"{ori}를 {sub}로 대체한 {recipe}입니다."),
            "ingredients": updated_recipe.get("ingredients", []),
            "instructions": updated_recipe.get("instructions", []),
            "substituteFailure": False,  # 명시적 성공 플래그
            "substitutionInfo": {
                "original": ori,
                "substitute": sub,
                "estimatedAmount": updated_recipe.get("estimatedAmount", "적당량"),
                "substitutionReason": updated_recipe.get("substitutionReason", "")
            }
        }

        # 🔧 상세 응답 로깅 추가
        print("=" * 50)
        print("📤 FLASK 응답 데이터 상세 정보")
        print("=" * 50)
        print(f"이름: {response_json['name']}")
        print(f"설명: {response_json['description']}")
        print(f"재료 개수: {len(response_json['ingredients'])}")
        
        print("\n📋 재료 목록:")
        for i, ingredient in enumerate(response_json['ingredients']):
            print(f"  {i+1}. {ingredient.get('name', 'N/A')}: {ingredient.get('amount', 'N/A')}")
        
        print(f"\n📝 조리법 개수: {len(response_json['instructions'])}")
        print("\n📝 조리법 목록:")
        for i, instruction in enumerate(response_json['instructions']):
            print(f"  단계 {instruction.get('stepNumber', i+1)}: {instruction.get('instruction', 'N/A')[:60]}...")
            print(f"    ⏰ 조리시간: {instruction.get('cookingTime', 0)}분 ({instruction.get('cookingTimeSeconds', 0)}초)")
        
        print(f"\n🔄 대체 정보:")
        print(f"  원재료: {response_json['substitutionInfo']['original']}")
        print(f"  대체재료: {response_json['substitutionInfo']['substitute']}")
        print(f"  권장 수량: {response_json['substitutionInfo']['estimatedAmount']}")
        
        print(f"\n✅ 대체 실패 여부: {response_json['substituteFailure']}")
        print("=" * 50)

        # JSON 직렬화 테스트
        import json
        try:
            json_str = json.dumps(response_json, ensure_ascii=False, indent=2)
            print("✅ JSON 직렬화 성공")
            print(f"JSON 크기: {len(json_str)} 문자")
        except Exception as json_error:
            print(f"❌ JSON 직렬화 실패: {json_error}")
            return jsonify({
                "error": f"JSON 직렬화 오류: {str(json_error)}",
                "substituteFailure": True
            }), 500

        print(f"대체 레시피 업데이트 성공: {ori} -> {sub}")
        return jsonify(response_json), 200

    except Exception as e:
        print(f"전체 처리 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": f"요청 처리 중 오류 발생: {str(e)}",
            "substituteFailure": True
        }), 500
    
# app.py의 evaluate_substitute_with_llm 함수 수정
def evaluate_substitute_with_llm(original_ingredient, substitute_ingredient, recipe_name, original_recipe=None):
    """
    LLM을 사용하여 대체 가능성을 평가하고 대체 레시피를 생성
    """
    try:
        # 원본 레시피 데이터를 문자열로 변환
        recipe_context = ""
        if original_recipe:
            recipe_context = "기존 레시피 정보:\n"
            
            # 재료 정보 추가
            if "ingredients" in original_recipe and original_recipe["ingredients"]:
                recipe_context += "재료:\n"
                for ing in original_recipe["ingredients"]:
                    name = ing.get("name", "")
                    amount = ing.get("amount", "적당량")
                    recipe_context += f"- {name}: {amount}\n"
            
            # 조리법 정보 추가
            if "instructions" in original_recipe and original_recipe["instructions"]:
                recipe_context += "\n조리법:\n"
                for i, inst in enumerate(original_recipe["instructions"]):
                    instruction = inst.get("instruction", "")
                    recipe_context += f"{i+1}. {instruction}\n"
        
        # LLM에 보낼 프롬프트 구성
        prompt = f"""
        당신은 요리 전문가입니다. 레시피에서 재료 대체 가능성을 판단하고 대체 레시피를 생성해야 합니다.
        
        레시피: {recipe_name}
        원재료: {original_ingredient}
        대체재료: {substitute_ingredient}
        
        {recipe_context}
        
        다음 단계를 수행하세요:
        
        1. 먼저 {original_ingredient}를 {substitute_ingredient}로 대체할 수 있는지 판단하세요. 
           - 맛, 식감, 조리 방법, 영양소 등을 고려하세요.
           - 대체 시 예상되는 결과와 영향을 분석하세요.
        
        2. 대체 가능성 판단 결과를 다음 형식으로 제공하세요:
           - 대체 가능: [가능/불가능]
           - 이유: [대체 가능/불가능한 상세 이유]
           - 권장 수량: [원재료 대비 적절한 대체재료 수량]
        
        3. 대체 가능하다고 판단되면 새로운 레시피를 생성하세요:
           - name: [대체 재료를 반영한 레시피 이름]
           - description: [대체 재료를 사용한 레시피 설명]
           - ingredients:
             * [재료1]: [수량]
             * [재료2]: [수량]
             ...
           - instructions:
             ### 1단계 ###
             [첫 번째 조리 단계]
             ### 2단계 ###
             [두 번째 조리 단계]
             ...
        
        4. 대체 불가능하다고 판단되면 왜 불가능한지 구체적인 이유를 설명하세요.
        
        JSON 형식으로 응답하지 말고, 요청한 형식으로만 정확히 답변하세요.
        """
        
        # LLM에 요청 보내기
        result = qa_chain.invoke({"question": prompt})
        response_text = result["answer"]
        print(f"LLM 응답:\n{response_text}")
        
        # response_lower 변수 정의
        response_lower = response_text.lower()
        
        # 부정적인 대체 표현 검사
        negative_indicators = [
            "대체할 수 없",
            "대체가 불가능",
            "적절하지 않",
            "권장하지 않",
            "사용하지 않는 것이 좋",
            "대체 불가능",
            "불가능합니다",
            "어렵습니다",
            "맞지 않습니다"
        ]
        
        found_negative = False
        for indicator in negative_indicators:
            if indicator.lower() in response_lower:
                found_negative = True
                print(f"부정 표현 발견: '{indicator}'")
                break
        
        # 응답 확인
        # 레시피를 제대로 생성했는지 확인 - 정규 표현식 패턴으로 변경
        has_recipe_format = False
        
        # 다양한 레시피 형식 패턴 확인
        recipe_patterns = [
            # 표준 형식
            r'name\s*:',
            r'description\s*:',
            r'ingredients\s*:',
            r'instructions\s*:',
            
            # 마크다운 형식
            r'\*\s*name\s*:',
            r'\*\s*description\s*:',
            r'\*\s*ingredients\s*:',
            r'\*\s*instructions\s*:',
            
            # 볼드 마크다운 형식
            r'\*\*\s*name\s*:\s*\*\*',
            r'\*\*\s*description\s*:\s*\*\*',
            r'\*\*\s*ingredients\s*:\s*\*\*',
            r'\*\*\s*instructions\s*:\s*\*\*'
        ]
        
        # 최소 2개 이상의 레시피 형식 패턴이 발견되면 유효한 레시피로 간주
        pattern_count = 0
        for pattern in recipe_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                pattern_count += 1
        
        has_recipe_format = pattern_count >= 2
        
        # 또는 "단계" 패턴이 여러 개 발견되면 유효한 레시피로 간주
        step_patterns = re.findall(r'###\s*\d+단계\s*###', response_text)
        if len(step_patterns) >= 3:  # 최소 3단계 이상
            has_recipe_format = True
        
        # 긍정적인 대체 가능 표현 검사
        positive_indicators = [
            "대체 가능합니다",
            "대체할 수 있습니다",
            "사용해도 됩니다",
            "문제 없습니다",
            "적합합니다",
            "좋은 대체재입니다",
            "충분히 가능합니다",
            "대체 가능성 분석",
            "대체 가능:"
        ]
        
        has_positive_indicator = False
        for indicator in positive_indicators:
            if indicator.lower() in response_lower:
                has_positive_indicator = True
                print(f"긍정 표현 발견: '{indicator}'")
                break
        
        # 첫 문단이 "네", "예"로 시작하는지 확인
        first_paragraph = response_text.split('\n')[0].strip().lower()
        starts_positive = first_paragraph.startswith('네') or first_paragraph.startswith('예') or '대체 가능' in first_paragraph
        
        # 최종 판단 - 대체 가능성 판단 로직
        # 1. 명시적 부정 표현이 없고
        # 2. (레시피 형식이 있거나 긍정 표현이 있거나 첫 문단이 긍정적으로 시작)
        is_possible = not found_negative and (has_recipe_format or has_positive_indicator or starts_positive)
        
        # 디버깅 정보
        first_line = response_text.split('\n')[0][:70] if response_text else ""
        print(f"LLM 응답 첫 문장: {first_line}...")
        print(f"레시피 형식 확인 결과: {has_recipe_format}")
        print(f"긍정 표현 확인 결과: {has_positive_indicator}")
        print(f"긍정적 시작 확인 결과: {starts_positive}")
        print(f"대체 가능 여부 최종 판단 결과: {is_possible}")
            
        if is_possible:
            print(f"LLM 대체 가능 판단: {original_ingredient}를 {substitute_ingredient}로 대체할 수 있습니다.")
            
            # 대체 가능한 경우 레시피 파싱
            recipe_data = {}
            
            # 🔧 수정된 레시피 이름 추출
            name_patterns = [
                r"\*\*name:\*\*\s*(.+?)(?=\n|$)",  # **name:** 패턴
                r"name\s*:\s*(.+?)(?=\n|$)",
                r"이름\s*:\s*(.+?)(?=\n|$)",
                r"\*\s*name\s*:\s*(.+?)(?=\n|$)",
                r"- name\s*:\s*(.+?)(?=\n|$)"
            ]
            
            name_found = False
            for pattern in name_patterns:
                name_match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                if name_match:
                    recipe_data["name"] = name_match.group(1).strip()
                    name_found = True
                    print(f"레시피 이름 추출됨: {recipe_data['name']}")
                    break
            
            if not name_found:
                recipe_data["name"] = f"{substitute_ingredient}를 사용한 {recipe_name}"
                print(f"기본 레시피 이름 사용: {recipe_data['name']}")
            
            # 🔧 수정된 설명 추출
            desc_patterns = [
                r"\*\*description:\*\*\s*(.+?)(?=\n|$)",  # **description:** 패턴
                r"description\s*:\s*(.+?)(?=\n|$)",
                r"설명\s*:\s*(.+?)(?=\n|$)",
                r"\*\s*description\s*:\s*(.+?)(?=\n|$)",
                r"- description\s*:\s*(.+?)(?=\n|$)"
            ]
            
            desc_found = False
            for pattern in desc_patterns:
                desc_match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                if desc_match:
                    recipe_data["description"] = desc_match.group(1).strip()
                    desc_found = True
                    print(f"레시피 설명 추출됨: {recipe_data['description']}")
                    break
            
            if not desc_found:
                recipe_data["description"] = f"{original_ingredient}를 {substitute_ingredient}로 대체한 {recipe_name}입니다."
                print(f"기본 레시피 설명 사용: {recipe_data['description']}")
            
            # 🔧 수정된 재료 추출 - 다양한 패턴 지원
            ingredients = []

            print(f"\n🔍 재료 추출 시작")
            print(f"전체 응답 텍스트 길이: {len(response_text)}")

            # 재료 섹션 추출 - 여러 패턴 시도
            ingredients_patterns = [
                r"\*\*ingredients:\*\*\s*(.*?)(?=\*\*instructions:|\*\*조리법:|조리법|instructions|\Z)",  # **ingredients:** 패턴
                r"ingredients\s*:\s*(.*?)(?=instructions|조리법|만드는법|\Z)",
                r"재료\s*:\s*(.*?)(?=조리법|만드는법|instructions|\Z)",
                r"- ingredients\s*:(.*?)(?=- instructions|조리법|\Z)"
            ]

            ingredients_text = ""
            used_pattern = ""

            for i, pattern in enumerate(ingredients_patterns):
                ingredients_match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
                if ingredients_match:
                    ingredients_text = ingredients_match.group(1).strip()
                    used_pattern = f"패턴 {i+1}"
                    print(f"재료 섹션 추출 성공 ({used_pattern}): 길이 {len(ingredients_text)}")
                    print(f"추출된 재료 텍스트 미리보기: {ingredients_text[:200]}...")
                    break

            if not ingredients_text:
                print("⚠️ 재료 섹션을 찾지 못했습니다. 전체 텍스트에서 재료 패턴 검색을 시도합니다.")
                # 전체 텍스트에서 재료 같은 패턴 찾기
                lines = response_text.split('\n')
                ingredient_section_started = False
                temp_ingredients = []
                
                for line in lines:
                    line = line.strip()
                    # 재료 섹션 시작 감지
                    if any(keyword in line.lower() for keyword in ['ingredients', '재료', '* 밥', '* 계란']):
                        ingredient_section_started = True
                        if line.startswith('*') and ':' in line:
                            temp_ingredients.append(line)
                        continue
                    
                    # 재료 섹션이 시작된 후 재료 라인 수집
                    if ingredient_section_started:
                        if line.startswith('*') and ':' in line:
                            temp_ingredients.append(line)
                        elif line.startswith('- instructions') or line.startswith('### ') or '단계' in line:
                            break  # 조리법 섹션 시작되면 중단
                        elif line and not line.startswith('*') and not line.startswith('-'):
                            # 다른 섹션이 시작되면 중단
                            break
                
                if temp_ingredients:
                    ingredients_text = '\n'.join(temp_ingredients)
                    print(f"전체 텍스트에서 재료 추출 성공: {len(temp_ingredients)}개 라인")
                    print(f"추출된 재료 텍스트: {ingredients_text}")

            if ingredients_text:
                print(f"\n📋 재료 파싱 시작")
                
                # 재료 라인별 파싱 - 다양한 패턴 지원 및 강화
                ingredient_patterns = [
                    r"\*\s*([^:]+?)\s*:\s*(.+?)(?=\n|$)",     # * 재료: 수량
                    r"([^:]+?)\s*:\s*(.+?)(?=\n|$)",          # 재료: 수량  
                    r"-\s*([^:]+?)\s*:\s*(.+?)(?=\n|$)",      # - 재료: 수량
                    r"\*\s*([^:]+?)\s+(.+?)(?=\n|$)"          # * 재료 수량 (콜론 없는 패턴)
                ]
                
                print(f"재료 파싱 대상 텍스트:\n{ingredients_text}")
                
                for pattern_idx, pattern in enumerate(ingredient_patterns):
                    ingredient_matches = re.findall(pattern, ingredients_text, re.MULTILINE)
                    print(f"패턴 {pattern_idx + 1} 시도: {len(ingredient_matches)}개 매치")
                    
                    if ingredient_matches:
                        print(f"✅ 패턴 {pattern_idx + 1} 성공!")
                        for match_idx, (ing_name, ing_amount) in enumerate(ingredient_matches):
                            # 이름과 수량 정리
                            clean_name = ing_name.strip().replace("*", "").replace("-", "").strip()
                            clean_amount = ing_amount.strip()
                            
                            print(f"  매치 {match_idx + 1}: '{ing_name}' : '{ing_amount}'")
                            print(f"  정리 후: '{clean_name}' : '{clean_amount}'")
                            
                            # 빈 값이나 잘못된 값 제외
                            if (clean_name and clean_amount and 
                                len(clean_name) > 0 and len(clean_amount) > 0 and
                                clean_name != clean_amount):  # 이름과 수량이 같으면 제외
                                
                                ingredients.append({
                                    "name": clean_name,
                                    "amount": clean_amount
                                })
                                print(f"  ✅ 재료 추가됨: {clean_name} - {clean_amount}")
                            else:
                                print(f"  ❌ 재료 제외됨 (빈 값 또는 중복): '{clean_name}' - '{clean_amount}'")
                        break
                
                print(f"\n📋 추출된 재료 최종 결과: {len(ingredients)}개")
                for i, ing in enumerate(ingredients):
                    print(f"  {i+1}. {ing['name']}: {ing['amount']}")
            else:
                print("❌ 재료 텍스트를 전혀 찾지 못했습니다.")

            # 재료가 여전히 비어있는 경우 기본 재료 생성
            if not ingredients:
                print("⚠️ 재료 추출 실패, 기본 재료 생성")
                
                # LLM 응답에서 언급된 재료들을 찾아서 기본 재료 생성
                default_ingredients_found = []
                
                # 일반적인 재료 키워드 검색
                common_ingredients = {
                    '계란': '2개',
                    '밥': '1공기', 
                    '들기름': '1큰술',
                    '간장': '1작은술',
                    '설탕': '1/2작은술',
                    '소금': '약간',
                    '쪽파': '약간',
                    '깨소금': '약간'
                }
                
                for ingredient_name, default_amount in common_ingredients.items():
                    if ingredient_name in response_text:
                        default_ingredients_found.append({
                            "name": ingredient_name,
                            "amount": default_amount
                        })
                        print(f"  기본 재료 추가: {ingredient_name} - {default_amount}")
                
                if default_ingredients_found:
                    ingredients = default_ingredients_found
                    print(f"기본 재료 {len(ingredients)}개 생성됨")
                else:
                    # 최후의 수단: 대체 재료만이라도 추가
                    ingredients = [{
                        "name": substitute_ingredient,
                        "amount": "적당량"
                    }]
                    print(f"최소 재료 생성: {substitute_ingredient}")

            recipe_data["ingredients"] = ingredients
            print(f"\n✅ 최종 재료 설정 완료: {len(ingredients)}개")
            
            # 🔧 완전히 개선된 조리법 추출
            instructions = []
            
            # 조리법 섹션 추출 - 다양한 패턴 시도
            instructions_patterns = [
                r"\*\*instructions:\*\*\s*(.*?)(?=\Z)",  # **instructions:** 패턴
                r"instructions\s*:\s*(.*?)(?=\Z)",
                r"조리법\s*:\s*(.*?)(?=\Z)",
                r"만드는 법\s*:\s*(.*?)(?=\Z)"
            ]
            
            instructions_text = ""
            for pattern in instructions_patterns:
                instructions_match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
                if instructions_match:
                    instructions_text = instructions_match.group(1).strip()
                    print(f"조리법 섹션 추출됨 (길이: {len(instructions_text)})")
                    break
            
            # 조리법 섹션을 찾지 못한 경우 전체 텍스트에서 단계 추출
            if not instructions_text:
                instructions_text = response_text
                print("전체 텍스트에서 조리법 단계 추출 시도")
            
            # 단계별 파싱 - 개선된 정규표현식
            step_patterns = [
                r"###\s*(\d+)단계\s*###\s*(.*?)(?=###\s*\d+단계\s*###|\Z)",  # ### N단계 ### 패턴
                r"(\d+)단계[:\s]*(.*?)(?=\d+단계|\Z)",                      # N단계: 패턴
                r"(\d+)\.\s*(.*?)(?=\d+\.|\Z)"                             # N. 패턴
            ]
            
            for pattern in step_patterns:
                step_matches = re.findall(pattern, instructions_text, re.DOTALL)
                if step_matches:
                    print(f"단계 패턴 매칭 성공: {len(step_matches)}개 단계 발견")
                    
                    for step_num, step_content in step_matches:
                        # 내용 정리
                        clean_content = step_content.strip()
                        
                        # 불필요한 마크다운 제거
                        clean_content = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_content)  # **텍스트** -> 텍스트
                        clean_content = re.sub(r'\n+', ' ', clean_content)  # 여러 줄바꿈을 공백으로
                        clean_content = clean_content.strip()
                        
                        if clean_content and len(clean_content) > 5:  # 의미있는 내용만
                            # 조리 시간 추출
                            cooking_time_mins, cooking_time_seconds = extract_cooking_time(clean_content)
                            
                            instructions.append({
                                "instruction": clean_content,
                                "cookingTime": cooking_time_mins,
                                "cookingTimeSeconds": cooking_time_seconds,
                                "stepNumber": int(step_num)
                            })
                            
                            print(f"단계 {step_num} 추가됨: {clean_content[:50]}...")
                    break
            
            # 여전히 조리법이 없는 경우 줄 단위로 분석
            if not instructions:
                print("패턴 매칭 실패, 줄 단위 분석 시도")
                lines = instructions_text.split('\n')
                step_counter = 1
                
                for line in lines:
                    line = line.strip()
                    # 의미있는 조리 단계로 보이는 라인만 추출
                    if (len(line) > 10 and 
                        ('단계' in line or '넣고' in line or '끓' in line or '굽' in line or '볶' in line or '섞' in line) and
                        not line.startswith('##') and not line.startswith('**')):
                        
                        # 단계 번호 제거
                        clean_line = re.sub(r'^\d+\.\s*|###\s*\d+단계\s*###\s*', '', line).strip()
                        
                        if clean_line:
                            cooking_time_mins, cooking_time_seconds = extract_cooking_time(clean_line)
                            
                            instructions.append({
                                "instruction": clean_line,
                                "cookingTime": cooking_time_mins,
                                "cookingTimeSeconds": cooking_time_seconds,
                                "stepNumber": step_counter
                            })
                            
                            print(f"라인 단계 {step_counter} 추가됨: {clean_line[:50]}...")
                            step_counter += 1
            
            print(f"최종 추출된 조리법 단계: {len(instructions)}개")
            for i, inst in enumerate(instructions):
                print(f"단계 {inst['stepNumber']}: {inst['instruction'][:70]}...")
            
            recipe_data["instructions"] = instructions
            
            # 대체 수량 및 이유 추출
            amount_match = re.search(r"권장 수량:\s*(.+?)(?=\n|$)", response_text, re.MULTILINE)
            if amount_match:
                recipe_data["estimatedAmount"] = amount_match.group(1).strip()
            
            reason_match = re.search(r"이유:\s*(.+?)(?=\n|$)", response_text, re.MULTILINE)
            if reason_match:
                recipe_data["substitutionReason"] = reason_match.group(1).strip()
            
            return True, recipe_data
        else:
            print(f"LLM 대체 불가능 판단: {original_ingredient}를 {substitute_ingredient}로 대체하는 것은 적절하지 않습니다.")
            # 대체 불가능한 경우
            reason = ""
            reason_match = re.search(r"이유:\s*(.+?)(?=$|\n)", response_text)
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                # 이유가 명시적으로 표시되지 않은 경우 전체 응답에서 관련 부분 추출
                impossible_section = re.search(r"대체 불가능.*", response_text, re.DOTALL)
                if impossible_section:
                    reason = impossible_section.group(0)
                else:
                    reason = f"{original_ingredient}를 {substitute_ingredient}로 대체하는 것은 권장되지 않습니다."
            
            return False, {"reason": reason}
    
    except Exception as e:
        print(f"LLM 평가 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, {"reason": f"평가 중 오류가 발생했습니다: {str(e)}"}
    
def update_ingredients_with_substitute(original_ingredients, ori, sub):
    """
    기존 재료 리스트에서 원재료를 대체재료로 교체 (강화된 버전)
    """
    updated_ingredients = []
    substitute_found = False
    
    for ingredient in original_ingredients:
        ingredient_name = ingredient.get("name", "").lower()
        ori_lower = ori.lower()
        
        # 원재료와 일치하는지 확인 (부분 일치 및 정확 일치)
        is_match = (
            ori_lower == ingredient_name or  # 정확 일치
            ori_lower in ingredient_name or  # 원재료가 재료명에 포함
            ingredient_name in ori_lower or  # 재료명이 원재료에 포함
            # 공백 및 특수문자 제거 후 비교
            ori_lower.replace(" ", "").replace("-", "") == ingredient_name.replace(" ", "").replace("-", "")
        )
        
        if is_match:
            # 대체 재료로 교체
            updated_ingredient = {
                "name": sub,
                "amount": ingredient.get("amount", "적당량")  # 기존 수량 유지
            }
            updated_ingredients.append(updated_ingredient)
            substitute_found = True
            print(f"재료 대체됨: {ingredient.get('name')} -> {sub}")
        else:
            # 기존 재료 유지
            updated_ingredients.append(ingredient.copy())
    
    # 원재료를 찾지 못한 경우 대체재료 추가
    if not substitute_found:
        print(f"원재료 '{ori}'를 찾지 못해 대체재료 '{sub}' 추가")
        updated_ingredients.append({
            "name": sub,
            "amount": "적당량"
        })
    
    return updated_ingredients


def update_instructions_with_substitute(original_instructions, ori, sub):
    """
    기존 조리법에서 원재료 언급을 대체재료로 교체 (강화된 정규표현식 사용)
    """
    updated_instructions = []
    
    for instruction in original_instructions:
        instruction_text = instruction.get("instruction", "")
        
        if not instruction_text:
            updated_instructions.append(instruction.copy())
            continue
        
        updated_text = instruction_text
        
        try:
            # 1. 정확히 일치하는 경우 (단어 경계 사용)
            import re
            exact_pattern = r'(?i)\b' + re.escape(ori) + r'\b'
            updated_text = re.sub(exact_pattern, sub, updated_text)
            
            # 2. 부분 일치하는 경우 (예: "무염버터" -> "무염마가린")
            if ori != sub and ori.lower() in updated_text.lower():
                partial_pattern = r'(?i)' + re.escape(ori)
                # 이미 대체되지 않았고, 대체재료가 포함되지 않은 경우에만 교체
                if ori.lower() in updated_text.lower() and sub.lower() not in updated_text.lower():
                    updated_text = re.sub(partial_pattern, sub, updated_text)
            
            # 3. 공백이나 하이픈이 포함된 재료명 처리
            ori_variants = [
                ori,
                ori.replace(" ", ""),
                ori.replace("-", ""),
                ori.replace("_", "")
            ]
            
            for variant in ori_variants:
                if variant != ori and variant.lower() in updated_text.lower():
                    variant_pattern = r'(?i)\b' + re.escape(variant) + r'\b'
                    updated_text = re.sub(variant_pattern, sub, updated_text)
            
        except Exception as regex_error:
            print(f"정규표현식 처리 오류 (원본 유지): {regex_error}")
            updated_text = instruction_text
        
        # 업데이트된 조리법 저장
        updated_instruction = instruction.copy()
        updated_instruction["instruction"] = updated_text
        updated_instructions.append(updated_instruction)
        
        if updated_text != instruction_text:
            print(f"조리법 업데이트됨: '{ori}' -> '{sub}'")
            print(f"  원본: {instruction_text[:50]}...")
            print(f"  수정: {updated_text[:50]}...")
    
    return updated_instructions


def estimate_substitute_amount(ori, sub, ingredients_list):
    """
    LLM을 사용하여 대체 재료의 적절한 수량 추정 (개선된 버전)
    """
    try:
        # 기존 재료에서 원재료의 수량 찾기
        original_amount = "적당량"
        for ingredient in ingredients_list:
            ingredient_name = ingredient.get("name", "").lower()
            if ori.lower() in ingredient_name or ingredient_name in ori.lower():
                original_amount = ingredient.get("amount", "적당량")
                break
        
        # LLM에게 대체 재료 수량 추정 요청
        query = f"""
        요리에서 '{ori}' {original_amount}를 '{sub}'로 대체할 때 적절한 수량을 알려주세요.
        
        답변은 오직 수량만 간단히 답해주세요. 예: "2큰술", "100g", "1개", "적당량"
        설명이나 부가적인 내용은 포함하지 마세요.
        
        중요: 재료의 밀도, 단맛, 짠맛 등의 특성 차이를 고려해주세요.
        """
        
        result = qa_chain.invoke({"question": query})
        estimated_amount = result["answer"].strip()
        
        # 응답에서 수량 부분만 추출 (개선된 정규표현식)
        import re
        amount_patterns = [
            r'(\d+(?:\.\d+)?\s*(?:큰술|작은술|컵|개|g|kg|ml|l|조각|편|대|뿌리|적당량|소량|약간))',
            r'(적당량|소량|약간)',
            r'(\d+(?:\.\d+)?)\s*(큰술|작은술|컵|개|g|kg|ml|l|조각|편|대|뿌리)'
        ]
        
        for pattern in amount_patterns:
            amount_match = re.search(pattern, estimated_amount, re.IGNORECASE)
            if amount_match:
                extracted_amount = amount_match.group(1) if len(amount_match.groups()) == 1 else f"{amount_match.group(1)}{amount_match.group(2)}"
                print(f"LLM 수량 추정 성공: {ori} {original_amount} -> {sub} {extracted_amount}")
                return extracted_amount
        
        # 패턴 매칭 실패 시 기본값 반환
        if "적당량" in estimated_amount or "소량" in estimated_amount or "약간" in estimated_amount:
            return "적당량"
        else:
            print(f"LLM 수량 추정 실패, 원본 수량 사용: {original_amount}")
            return original_amount  # 추정 실패 시 원래 수량 사용
            
    except Exception as e:
        print(f"수량 추정 오류: {str(e)}")
        return "적당량"
    
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

        print(f"\n🥗 영양정보 요청 받음")
        print(f"입력 재료: {ingredients}")
        
        response_text = get_nutrition_info(ingredients)
        print(f"\n🧠 LLM 응답 원본:")
        print(f"응답 길이: {len(response_text)}")
        print(f"응답 내용:\n{response_text}")
        
        if not response_text:
            print("❌ LLM 응답이 비어있음")
            return jsonify({"error": "모델 응답이 비었습니다."}), 500

        # 영양정보 추출
        result = extract_nutrition(response_text)
        
        print(f"\n📊 최종 API 응답:")
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 검증: 주요 영양소가 모두 0인지 확인
        major_nutrients = ['calories', 'carbohydrate', 'protein', 'fat']
        zero_count = sum(1 for nutrient in major_nutrients if result.get(nutrient, 0) == 0)
        
        if zero_count >= 3:
            print(f"⚠️ 경고: 주요 영양소 {zero_count}개가 0값입니다.")
            print("기본값으로 보정된 응답을 반환합니다.")
        
        return jsonify(result)

    except Exception as e:
        print(f"❌ 영양정보 처리 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
def extract_nutrition(text):
    def extract_value(label, default=0.0):
        # 마크다운 별표(*) 및 대시(-) 제거
        clean_text = re.sub(r'^\s*\*\s*|\*\*', '', text, flags=re.MULTILINE)
        
        print(f"\n🔍 '{label}' 추출 시작")
        print(f"정리된 텍스트 샘플: {clean_text[:200]}...")
        
        # 🔧 개선된 라벨 패턴 - 더 유연한 매칭
        label_patterns = [
            # 기본 패턴: - 라벨: 값
            rf'[-*]?\s*{re.escape(label)}\s*:\s*(?:약\s*)?(.*?)(?:\n|$)',
            # 마크다운 패턴: **라벨**: 값  
            rf'\*\*{re.escape(label)}\*\*\s*:\s*(?:약\s*)?(.*?)(?:\n|$)',
            # 공백 포함 패턴
            rf'[-*]?\s*{re.escape(label)}\s+(?:약\s*)?(.*?)(?:\n|$)',
            # 콜론 없는 패턴
            rf'{re.escape(label)}\s+(?:약\s*)?([\d.,]+\s*\w+)',
        ]
        
        full_value = None
        used_pattern = ""
        
        for i, pattern in enumerate(label_patterns):
            match = re.search(pattern, clean_text, re.IGNORECASE | re.MULTILINE)
            if match:
                full_value = match.group(1).strip()
                used_pattern = f"패턴 {i+1}"
                print(f"✅ {used_pattern} 매칭 성공: '{full_value}'")
                break
        
        if not full_value:
            print(f"❌ '{label}' 패턴 매칭 실패")
            return default
        
        print(f"라벨 '{label}'에 대한 추출된 전체 값: {full_value}")
        
        # 설명 부분 제거 (괄호 안 내용)
        value_without_desc = re.sub(r'\s*\(.*?\)', '', full_value)
        print(f"설명 제거 후 값: {value_without_desc}")
        
        # 🔧 개선된 숫자 추출 로직
        try:
            # 1. "미량", "0" 등의 특수 케이스 먼저 처리
            if any(keyword in value_without_desc.lower() for keyword in ['미량', '없음', 'trace']):
                print(f"특수 케이스 감지: 0.0 반환")
                return 0.0
            
            # 2. 범위 값 처리 (예: "450-600kcal", "15~20g")
            range_pattern = r'(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)'
            range_match = re.search(range_pattern, value_without_desc)
            if range_match:
                num1, num2 = float(range_match.group(1)), float(range_match.group(2))
                average = (num1 + num2) / 2
                print(f"범위 값 처리: {num1}-{num2} → 평균 {average}")
                return average
            
            # 3. 일반 숫자 추출 - 더 정확한 패턴
            # 소수점, 콤마가 포함된 숫자 매칭
            number_patterns = [
                r'(\d+(?:\.\d+)?)\s*(?:kcal|칼로리)',  # 칼로리 전용
                r'(\d+(?:\.\d+)?)\s*(?:mg|밀리그램)',   # mg 단위
                r'(\d+(?:\.\d+)?)\s*(?:g|그램)',       # g 단위  
                r'(\d+(?:\.\d+)?)',                     # 순수 숫자
            ]
            
            for pattern in number_patterns:
                num_match = re.search(pattern, value_without_desc)
                if num_match:
                    extracted_number = float(num_match.group(1))
                    print(f"숫자 추출 성공: {extracted_number}")
                    return extracted_number
            
            print(f"숫자 추출 실패, 기본값 반환: {default}")
            return default
            
        except (ValueError, AttributeError) as e:
            print(f"숫자 변환 오류: {e}, 기본값 반환: {default}")
            return default

    print(f"\n🍎 영양정보 추출 시작")
    print(f"입력 텍스트 길이: {len(text)}")
    print(f"입력 텍스트 미리보기:\n{text[:300]}...")

    # 각 영양소에 대해 라벨 기반 추출 수행
    result = {
        "calories": extract_value("칼로리"),
        "carbohydrate": extract_value("탄수화물"),
        "protein": extract_value("단백질"),
        "fat": extract_value("지방"),
        "sugar": extract_value("당"),
        "sodium": extract_value("나트륨"),
        "saturatedFat": extract_value("포화지방"),
        "transFat": extract_value("트랜스지방"),
        "cholesterol": extract_value("콜레스테롤")
    }
    
    # 🔧 추가 검증: 주요 영양소가 0인 경우 재시도
    if result["carbohydrate"] == 0.0 and "탄수화물" in text:
        print("⚠️ 탄수화물 재추출 시도")
        # 다른 표현으로 재시도
        alt_patterns = [r'탄수화물.*?(\d+(?:\.\d+)?)', r'탄수.*?(\d+(?:\.\d+)?)']
        for pattern in alt_patterns:
            match = re.search(pattern, text)
            if match:
                result["carbohydrate"] = float(match.group(1))
                print(f"탄수화물 재추출 성공: {result['carbohydrate']}")
                break
    
    if result["protein"] == 0.0 and "단백질" in text:
        print("⚠️ 단백질 재추출 시도")
        alt_patterns = [r'단백질.*?(\d+(?:\.\d+)?)', r'단백.*?(\d+(?:\.\d+)?)']
        for pattern in alt_patterns:
            match = re.search(pattern, text)
            if match:
                result["protein"] = float(match.group(1))
                print(f"단백질 재추출 성공: {result['protein']}")
                break
    
    # 디버깅을 위한 결과 로깅
    print(f"\n🎯 최종 추출된 영양 정보:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # 🔧 최종 검증: 모든 값이 0인 경우 기본값 설정
    non_zero_count = sum(1 for v in result.values() if v > 0)
    if non_zero_count < 3:  # 3개 미만의 영양소만 추출된 경우
        print("⚠️ 추출된 영양소가 너무 적음, 기본값 보정")
        if result["calories"] == 0:
            result["calories"] = 500.0
        if result["carbohydrate"] == 0:
            result["carbohydrate"] = 30.0
        if result["protein"] == 0:
            result["protein"] = 20.0
        if result["fat"] == 0:
            result["fat"] = 15.0
    
    print(f"\n✅ 최종 영양 정보 (보정 후):")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
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

        # 사용자 식습관 및 선호도 정보 가져오기
        user_habit = request.form.get('userHabit', '')
        user_preference = request.form.get('userPreference', '')
        
        # 인코딩 디버깅
        print(f"사용자 식습관 (repr): {user_habit!r}")
        print(f"사용자 선호도 (repr): {user_preference!r}")
        
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

            사용자 식습관: {user_habit}
            사용자 선호도: {user_preference}

            위 정보를 고려하여 사용자의 식습관과 선호도에 맞는 레시피를 생성해주세요.
            예를 들어, 사용자가 채식주의자라면 동물성 재료를 사용하지 않고,
            저탄수화물 식이를 선호한다면 탄수화물이 적은 레시피를 제공해주세요.

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
                  이미지에서 감지된 재료 중심으로 레시피를 제공해주세요.
            """
        else:
            combined_query = f"""
            {instructions}

            사용자 식습관: {user_habit}
            사용자 선호도: {user_preference}
            
            위 정보를 고려하여 사용자의 식습관과 선호도에 맞는 레시피를 생성해주세요.

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
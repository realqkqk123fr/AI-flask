# model/similarity_check.py
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1) 임베딩 데이터 로드
with open("FlavorGraph Node Embedding.json", "r", encoding="utf-8") as f:
    emb_dict = json.load(f)

# 문자열 키 → numpy 배열
food_emb = {name: np.array(vec, dtype=np.float32) for name, vec in emb_dict.items()}

# 2) 대체 가능 여부 판단 함수 → float 유사도 반환
def check_replace(orig: str, sub: str) -> float:
    try:
        vec_a = food_emb[orig]
        vec_b = food_emb[sub]
    except KeyError:
        raise ValueError("임베딩 데이터에 없는 재료입니다.")

    score = float(cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0, 0])
    return score

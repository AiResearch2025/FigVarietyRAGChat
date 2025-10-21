# === 특징 기반 벡터DB 성능 테스트 ===

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === 설정 ===
DB_PATH = "./features_db"
MODEL_NAME = "intfloat/multilingual-e5-base"

# === 임베딩 모델 로드 ===
embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# === 벡터DB 로드 ===
try:
    db = FAISS.load_local(DB_PATH, embedding, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"벡터DB를 로드하는 중 오류가 발생했습니다: {e}")
    print("먼저 build_features_db.py를 실행하여 DB를 생성했는지 확인하세요.")
    exit()

# === 테스트 쿼리 ===
test_queries = [
    "나무가 옆으로 퍼지면서 자라는 품종은?",
    "열매가 아주 큰 품종을 추천해줘.",
    "이탈리아가 원산지인 품종은 뭐야?"
]

print("🚀 특징 기반 벡터DB 성능 테스트를 시작합니다...")
print("-" * 40)

# === 각 쿼리에 대해 검색 수행 ===
for query in test_queries:
    # k=2로 설정하여 가장 유사한 결과 2개를 가져옵니다.
    results = db.similarity_search_with_score(query, k=2)
    
    print(f"❓ 질문: \"{query}\"")
    
    if results:
        print("✅ 검색 결과 (유사도 점수가 낮을수록 더 유사함):")
        for doc, score in results:
            variety = doc.metadata.get('variety', 'N/A')
            print(f"  - 품종: {variety:<20} | 내용: {doc.page_content} | 유사도: {score:.4f}")
    else:
        print("❌ 유사한 품종을 찾지 못했습니다.")
    
    print("-" * 40)

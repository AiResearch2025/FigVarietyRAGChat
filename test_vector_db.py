# === 벡터DB 성능 테스트 ===

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === 설정 ===
DB_PATH = "./vector_db"
MODEL_NAME = "intfloat/multilingual-e5-base"

# === 임베딩 모델 로드 ===
# build_faiss.py에서 사용한 것과 동일한 모델을 사용해야 합니다.
embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# === 벡터DB 로드 ===
# allow_dangerous_deserialization=True는 로컬 저장소에서 pickle 파일을 로드할 때 필요합니다.
db = FAISS.load_local(DB_PATH, embedding, allow_dangerous_deserialization=True)

# === 테스트 쿼리 ===
test_queries = [
    "SV는 어떤 환경에서 잘 자라나요?",
    "Ciccio Vero는 키우기 쉬운가요?",
    "제가 가지고 있는 무화과는 봉래시인가요?",
    # "당도 높은 무화과는?"
]

print("🚀 벡터DB 성능 테스트를 시작합니다...")
print("-" * 30)

# === 각 쿼리에 대해 검색 수행 ===
for query in test_queries:
    # k=1로 설정하여 가장 유사한 결과 1개만 가져옵니다.
    results = db.similarity_search(query, k=1)
    
    print(f"❓ 질문: \"{query}\"")
    
    if results:
        # 결과에서 품종 이름(source)을 추출합니다.
        retrieved_variety = results[0].metadata['source']
        print(f"✅ 가장 유사한 품종: {retrieved_variety}")
    else:
        print("❌ 유사한 품종을 찾지 못했습니다.")
    
    print("-" * 30)

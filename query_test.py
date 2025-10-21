from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from query_cases import QueryCase, classify_query

# === 설정 ===
DB_PATH = "./vector_db"
MODEL_NAME = "intfloat/multilingual-e5-base"

# === 임베딩 모델 로드 ===
embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# === 벡터DB 로드 ===
db = FAISS.load_local(DB_PATH, embedding, allow_dangerous_deserialization=True)

# === 테스트 쿼리 ===
queries = [
    "BNR은 당도가 높나요?",
    "Ciccio Vero에 대해 알려줘",
    "브런즈윅은 어떤 맛인가요?",
    "달콤한 무화과 품종 추천해줘",
    "이 사진 속 무화과는 무슨 품종이야?",
    "삽목이 잘 안 되는 품종 찾아줘"
]

# === 검색 ===
for query in queries:
    case = classify_query(query)
    print(f"--- Query: '{query}' ---")
    print(f"Case: {case.name}")

    # RAG 검색이 필요한 경우에만 수행 (예시)
    if case in [QueryCase.KNOWN_VARIETY_EXISTS, QueryCase.UNKNOWN_VARIETY_GENERAL_QUERY]:
        print("Performing RAG search...")
        docs = db.similarity_search(query, k=3)
        for d in docs:
            print(f"  📌 품종: {d.metadata['source']}")
    else:
        print("RAG search skipped for this case.")
    
    print("-" * (len(query) + 12), "\n")

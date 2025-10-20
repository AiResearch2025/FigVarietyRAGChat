from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === 설정 ===
DB_PATH = "./vector_db"
MODEL_NAME = "intfloat/multilingual-e5-base"

# === 임베딩 모델 로드 ===
embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# === 벡터DB 로드 ===
db = FAISS.load_local(DB_PATH, embedding, allow_dangerous_deserialization=True)

# === 검색 ===
query = "삽목이 잘 안 되는 품종 찾아줘"
docs = db.similarity_search(query, k=3)

for d in docs:
    print(f"📌 품종: {d.metadata['source']}")

# === 벡터DB 생성 ===

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  

# === 설정 ===
DATA_DIR = "./data"
DB_PATH = "./vector_db"
MODEL_NAME = "intfloat/multilingual-e5-base"

# === 임베딩 모델 로드 ===
embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# === 데이터 로드 ===
docs = []
for fname in os.listdir(DATA_DIR):
    if fname.endswith(".txt"):
        path = os.path.join(DATA_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            docs.append({"text": text, "source": fname})

# === 텍스트 분할 ===
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = []
metadatas = []

for doc in docs:
    chunks = splitter.split_text(doc["text"])
    for chunk in chunks:
        texts.append(chunk)
        metadatas.append({"source": doc["source"]})

# === FAISS 인덱스 생성 ===
vectorstore = FAISS.from_texts(texts, embedding=embedding, metadatas=metadatas)

# === 저장 ===
os.makedirs(DB_PATH, exist_ok=True)
vectorstore.save_local(DB_PATH)

print(f"✅ 벡터DB 저장 완료: {DB_PATH}")

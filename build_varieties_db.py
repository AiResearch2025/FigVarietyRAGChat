# === varieties 벡터DB 생성 ===

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === 설정 ===
DATA_DIR = "./varieties"
DB_PATH = "./vector_db"
MODEL_NAME = "intfloat/multilingual-e5-base"

# === 임베딩 모델 로드 ===
embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# === 데이터 생성 (파일 이름 기반) ===
texts = []
metadatas = []
for fname in os.listdir(DATA_DIR):
    if fname.endswith(".txt"):
        # 파일 이름에서 확장자를 제거하고 " fig variety"를 추가합니다.
        variety_name = os.path.splitext(fname)[0]
        text = f"{variety_name.replace('_', ' ')} fig variety"
        texts.append(text)
        metadatas.append({"source": fname})

print("📝 벡터 DB에 저장될 텍스트:")
for t in texts:
    print(f"- {t}")

# === FAISS 인덱스 생성 ===
vectorstore = FAISS.from_texts(texts, embedding=embedding, metadatas=metadatas)

# === 저장 ===
os.makedirs(DB_PATH, exist_ok=True)
vectorstore.save_local(DB_PATH)

print(f"\n✅ 벡터DB 저장 완료: {DB_PATH}")

# === 저장된 모든 데이터 확인 ===
print("\n🔍 벡터DB에 저장된 모든 단어 확인:")
# FAISS 인덱스에서 모든 문서를 가져오는 직접적인 방법은 없지만,
# 인덱스의 모든 벡터를 검색하여 내용을 확인할 수 있습니다.
# FAISS는 ID를 0부터 순차적으로 부여하므로, index_to_docstore_id를 통해 접근합니다.
ids = list(vectorstore.index_to_docstore_id.values())
retrieved_docs = vectorstore.docstore._dict
for i, doc_id in enumerate(ids):
    content = retrieved_docs[doc_id].page_content
    print(f"  {i+1}. {content} (Source: {retrieved_docs[doc_id].metadata['source']})")

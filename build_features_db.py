# === features 벡터DB 생성 ===

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === 설정 ===
FEATURES_DIR = "./features"  # 특징 파일이 있는 디렉토리
DB_PATH = "./features_db"   # 새로운 벡터DB를 저장할 경로
MODEL_NAME = "intfloat/multilingual-e5-base"

# === 임베딩 모델 로드 ===
embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# === 데이터 생성 (특징 파일 기반) ===
texts = []
metadatas = []

print("📝 특징 파일을 파싱하여 문장을 생성합니다...")

# ./result 디렉토리의 모든 .txt 파일을 순회합니다.
for fname in os.listdir(FEATURES_DIR):
    if not fname.endswith(".txt"):
        continue

    feature_category = os.path.splitext(fname)[0].replace('_', ' ')
    file_path = os.path.join(FEATURES_DIR, fname)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            
            # "특징 값: 품종1, 품종2, ..." 형식으로 분리합니다.
            feature_value, varieties_str = line.split(":", 1)
            feature_value = feature_value.strip()
            varieties = [v.strip() for v in varieties_str.split(",")]

            # 각 품종에 대해 문장을 생성합니다.
            for variety in varieties:
                if not variety: continue
                # 예: "브런즈윅 품종의 나무 수형은(는) 개장형입니다."
                text = f'"{variety}" 품종의 "{feature_category}"은(는) "{feature_value}"입니다.'
                texts.append(text)
                # 메타데이터에는 원본 파일명과 품종명을 저장하여 나중에 참조할 수 있도록 합니다.
                metadatas.append({"source_file": fname, "variety": variety})
                print(f"  - 생성: {text}")


# === FAISS 인덱스 생성 ===
if texts:
    vectorstore = FAISS.from_texts(texts, embedding=embedding, metadatas=metadatas)

    # === 저장 ===
    os.makedirs(DB_PATH, exist_ok=True)
    vectorstore.save_local(DB_PATH)

    print(f"\n✅ 특징 벡터DB 저장 완료: {DB_PATH}")

    # === 저장된 모든 데이터 확인 (옵션) ===
    print("\n🔍 특징 벡터DB에 저장된 모든 문장 확인:")
    ids = list(vectorstore.index_to_docstore_id.values())
    retrieved_docs = vectorstore.docstore._dict
    for i, doc_id in enumerate(ids):
        content = retrieved_docs[doc_id].page_content
        metadata = retrieved_docs[doc_id].metadata
        print(f"  {i+1}. {content} (Source: {metadata['source_file']}, Variety: {metadata['variety']})")
else:
    print("\n❌ 처리할 텍스트가 없어 벡터DB를 생성하지 않았습니다.")


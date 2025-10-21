# === Fig Variety AI Agent (v2) ===

import os
from enum import Enum, auto
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class QueryCategory(Enum):
    """사용자 질문의 의도를 나타내는 분류 Enum"""
    KNOWN_VARIETY_EXISTS = auto()
    KNOWN_VARIETY_TYPO = auto()
    KNOWN_VARIETY_NEEDS_CLARIFICATION = auto()
    UNKNOWN_VARIETY_GENERAL_QUERY = auto()
    UNKNOWN_VARIETY_IMAGE_QUERY = auto()
    CHITCHAT = auto()

class FigAgent:
    """
    사용자 질문을 분류하고 RAG 파이프라인을 관리하는 AI 에이전트 (개선 버전)
    """
    def __init__(self, varieties_db_path="./vector_db", features_db_path="./features_db", model_name="intfloat/multilingual-e5-base"):
        print("🤖 Fig Agent를 초기화하는 중입니다...")
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)
        
        try:
            self.varieties_db = FAISS.load_local(varieties_db_path, self.embedding, allow_dangerous_deserialization=True)
            self.features_db = FAISS.load_local(features_db_path, self.embedding, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"DB 로드 실패: {e}")
            raise

        # --- 별명 및 오타 처리 시스템 ---
        self.variety_aliases = {
            "Brunswick": ["brunswick", "브런즈윅"],
            "Ciccio_Nero": ["ciccio nero", "씨씨오 네로", "ciccio vero"],
            "Coll_De_Dama_Rimada": ["coll de dama rimada", "콜 드 다마 리마다"],
            "Hardy_Chicago": ["hardy chicago", "하디 시카고", "시카고"],
            "Horaishi": ["horaishi", "호래시", "봉래시"],
            "Strawberry_Verte": ["strawberry verte", "스트로베리 베르테", "sv"]
        }
        self.known_typos = ["ciccio vero"]
        
        # 긴 이름 우선 매칭을 위해 모든 별명을 길이순으로 정렬
        self.sorted_aliases = []
        for canonical, alias_list in self.variety_aliases.items():
            for alias in alias_list:
                self.sorted_aliases.append((alias, canonical))
        self.sorted_aliases.sort(key=lambda x: len(x[0]), reverse=True)
        
        print("✅ Agent 초기화 완료.")

    def _extract_variety_from_query(self, query: str) -> tuple[str, str] | None:
        """질문에서 (매칭된 별명, 표준 품종명) 튜플을 추출합니다."""
        query_lower = query.lower()
        for alias, canonical in self.sorted_aliases:
            if alias in query_lower:
                return (alias, canonical)
        return None

    def _classify_query(self, query: str) -> tuple[QueryCategory, dict]:
        """질문을 분석하여 카테고리와 관련 데이터를 반환합니다."""
        
        chitchat_keywords = ["안녕", "고마워", "감사", "땡큐"]
        if any(keyword in query for keyword in chitchat_keywords):
            return QueryCategory.CHITCHAT, {}

        image_keywords = ["사진", "이미지", "이 무화과"]
        if any(keyword in query for keyword in image_keywords):
            return QueryCategory.UNKNOWN_VARIETY_IMAGE_QUERY, {}

        variety_info = self._extract_variety_from_query(query)
        if variety_info:
            matched_alias, canonical_name = variety_info
            canonical_name_display = canonical_name.replace('_', ' ')

            if matched_alias in self.known_typos:
                return QueryCategory.KNOWN_VARIETY_TYPO, {"suggestion": canonical_name_display}

            clarification_keywords = ["어때", "좋아", "키울만", "괜찮"]
            if any(keyword in query for keyword in clarification_keywords):
                return QueryCategory.KNOWN_VARIETY_NEEDS_CLARIFICATION, {"variety": canonical_name_display}
            
            return QueryCategory.KNOWN_VARIETY_EXISTS, {"query": query, "variety": canonical_name_display}

        return QueryCategory.UNKNOWN_VARIETY_GENERAL_QUERY, {"query": query}

    def handle_query(self, query: str) -> str:
        """질문을 받아 분류하고, 각 케이스에 맞는 응답을 생성합니다."""
        category, data = self._classify_query(query)
        
        print(f"🧠 Agent 분류 결과: {category.name}")

        if category == QueryCategory.CHITCHAT:
            return "천만에요! 더 궁금한 점이 있으신가요?" if any(k in query for k in ["고마워", "감사", "땡큐"]) else "안녕하세요! 무화과에 대해 무엇이든 물어보세요."

        elif category == QueryCategory.UNKNOWN_VARIETY_IMAGE_QUERY:
            return "현재는 이미지를 분석하여 품종을 식별하는 기능이 지원되지 않습니다."

        elif category == QueryCategory.KNOWN_VARIETY_TYPO:
            return f"혹시 '{data['suggestion']}' 품종을 말씀하신 건가요?"

        elif category == QueryCategory.KNOWN_VARIETY_NEEDS_CLARIFICATION:
            return f"'{data['variety']}' 품종에 대해 어떤 점이 궁금하신가요? 맛, 생산성, 나무 크기 등 구체적인 기준을 알려주시면 더 자세히 답변해 드릴 수 있습니다."

        elif category == QueryCategory.KNOWN_VARIETY_EXISTS:
            print(f"  ▶️ '{data['variety']}'에 대한 정보 검색 중...")
            results = self.features_db.similarity_search(data['query'], k=3)
            if not results:
                return f"'{data['variety']}'에 대한 관련 정보를 찾지 못했습니다."
            
            context = "\n".join([f"- {doc.page_content}" for doc in results])
            return f"'{data['query']}'에 대한 검색 결과입니다:\n{context}"

        elif category == QueryCategory.UNKNOWN_VARIETY_GENERAL_QUERY:
            print("  ▶️ 특징 DB에서 추천 품종 검색 중...")
            results = self.features_db.similarity_search(data['query'], k=2)
            if not results:
                return "관련 품종을 찾지 못했습니다."

            context = "\n".join([f"- {doc.page_content}" for doc in results])
            return f"'{data['query']}'와 관련하여 다음 품종 정보를 찾았습니다:\n{context}"
            
        else:
            return "죄송합니다. 질문을 이해하지 못했습니다."

# === Agent 실행 예시 ===
if __name__ == '__main__':
    try:
        agent = FigAgent()
        
        queries_to_test = [
            "안녕?",
            "브런즈윅의 열매는 작은가요?",
            "Ciccio Vero는 키우기 쉬운가요?",
            "브런즈윅은 키울만 한가요?",
            "달콤한 품종 추천해줘",
            "이 사진 속 무화과는 무슨 품종이야?",
            "고마워",
            "안녕?"
        ]
        
        for q in queries_to_test:
            print(f"\n👤 사용자 질문: {q}")
            response = agent.handle_query(q)
            print(f"🤖 Agent 응답: {response}")

    except Exception as e:
        print(f"Agent 실행 중 오류 발생: {e}")

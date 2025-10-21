# === Fig Variety AI Agent (v3.2 - Gemini Hotfix) ===

import os
from enum import Enum, auto
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # Deprecation 경고 해결
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

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
    사용자 질문을 분류하고 RAG 파이프라인을 관리하는 AI 에이전트 (Gemini 버전)
    """
    def __init__(self, varieties_db_path="./vector_db", features_db_path="./features_db", model_name="intfloat/multilingual-e5-base"):
        print("🤖 Fig Agent (Gemini)를 초기화하는 중입니다...")
        
        # Deprecation 경고가 해결된 새로운 클래스를 사용합니다.
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)
        
        try:
            self.varieties_db = FAISS.load_local(varieties_db_path, self.embedding, allow_dangerous_deserialization=True)
            self.features_db = FAISS.load_local(features_db_path, self.embedding, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"DB 로드 실패: {e}")
            raise

        # --- LLM 및 RAG 체인 설정 (Gemini) ---
        # os.getenv를 사용하여 GOOGLE_API_KEY를 명시적으로 불러옵니다.
        google_api_key = os.getenv("GEMINI_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY 환경 변수를 찾을 수 없습니다. .env 파일을 확인해주세요.")
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", 
            google_api_key=google_api_key, 
            temperature=0, 
            convert_system_message_to_human=True
        )
        self.retriever = self.features_db.as_retriever(search_kwargs={'k': 3})
        
        prompt_template = """
        당신은 무화과 품종 전문가입니다. 사용자의 질문에 대해 아래의 '검색된 정보'를 바탕으로 친절하고 명확하게 답변해주세요.
        답변은 반드시 한국어로 작성해야 합니다. 정보가 부족하여 답변할 수 없는 경우, "정보가 부족하여 답변하기 어렵습니다."라고 솔직하게 말해주세요.
        
        [검색된 정보]
        {context}
        
        [사용자 질문]
        {question}
        
        [전문가 답변]
        """
        self.prompt = ChatPromptTemplate.from_template(prompt_template)

        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

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
            print(f"  ▶️ '{data['variety']}'에 대한 정보 검색 및 답변 생성 중...")
            return self.rag_chain.invoke(data['query'])

        elif category == QueryCategory.UNKNOWN_VARIETY_GENERAL_QUERY:
            print("  ▶️ 특징 DB에서 추천 품종 검색 및 답변 생성 중...")
            return self.rag_chain.invoke(data['query'])
            
        else:
            return "죄송합니다. 질문을 이해하지 못했습니다."

# === Agent 실행 예시 ===
if __name__ == '__main__':
    try:
        agent = FigAgent()
        
        queries_to_test = [
            "안녕?",
            "브런즈윅의 내한성은 어떤가요?",
            "Ciccio Vero는 키우기 쉬운가요?",
            "브런즈윅은 키울만 한가요?",
            "달콤한 품종 추천해줘",
            "이 사진 속 무화과는 무슨 품종이야?",
            "고마워"
        ]
        
        for q in queries_to_test:
            print(f"\n👤 사용자 질문: {q}")
            response = agent.handle_query(q)
            print(f"🤖 Agent 응답: {response}")

    except Exception as e:
        print(f"Agent 실행 중 오류 발생: {e}")
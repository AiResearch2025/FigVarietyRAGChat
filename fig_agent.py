# === Fig Variety AI Agent (v4.0 - Direct Gemini SDK) ===

import os
from enum import Enum, auto
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
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
    사용자 질문을 분류하고 RAG 파이프라인을 관리하는 AI 에이전트 (Gemini SDK 직접 호출 버전)
    """
    def __init__(self, varieties_db_path="./vector_db", features_db_path="./features_db", model_name="intfloat/multilingual-e5-base"):
        print("🤖 Fig Agent (Direct Gemini SDK)를 초기화하는 중입니다...")
        
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)
        
        try:
            self.varieties_db = FAISS.load_local(varieties_db_path, self.embedding, allow_dangerous_deserialization=True)
            self.features_db = FAISS.load_local(features_db_path, self.embedding, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"DB 로드 실패: {e}")
            raise

        # --- LLM 설정 (google-generativeai SDK 직접 사용) ---
        google_api_key = os.getenv("GEMINI_API_KEY")
        if not google_api_key:
            raise ValueError("GEMINI_API_KEY 환경 변수를 찾을 수 없습니다. .env 파일을 확인해주세요.")
        
        genai.configure(api_key=google_api_key)
        self.llm = genai.GenerativeModel('gemini-2.5-flash') 

        # --- Retriever 설정 (LangChain 사용) ---
        self.retriever = self.features_db.as_retriever(search_kwargs={'k': 3})
        
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

    def _generate_rag_response(self, query: str) -> str:
        """RAG 파이프라인을 수동으로 실행하여 답변을 생성합니다."""
        print(f"  🔍 '{query}'에 대한 관련 문서 검색 중...")
        docs = self.retriever.get_relevant_documents(query)

        if not docs:
            return "관련 정보를 찾지 못했습니다. 질문을 조금 더 구체적으로 해주시겠어요?"

        # 검색된 문서 내용을 하나의 컨텍스트 문자열로 합칩니다.
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt_template = f"""
        당신은 무화과 품종 전문가입니다. 사용자의 질문에 대해 아래의 '검색된 정보'를 바탕으로 친절하고 명확하게 답변해주세요.
        답변은 반드시 한국어로 작성해야 합니다. 정보가 부족하여 답변할 수 없는 경우, "정보가 부족하여 답변하기 어렵습니다."라고 솔직하게 말해주세요.
        
        [검색된 정보]
        {context}
        
        [사용자 질문]
        {query}
        
        [전문가 답변]
        """
        
        print("  🧠 LLM이 답변 생성 중...")
        try:
            response = self.llm.generate_content(prompt_template)
            return response.text
        except Exception as e:
            print(f"  ❗️ LLM 호출 중 오류 발생: {e}")
            return "답변을 생성하는 중에 오류가 발생했습니다."


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
            print(f"  ▶️ '{data['variety']}'에 대한 정보 검색 및 답변 생성 시작...")
            return self._generate_rag_response(data['query'])

        elif category == QueryCategory.UNKNOWN_VARIETY_GENERAL_QUERY:
            print("  ▶️ 특징 DB에서 추천 품종 검색 및 답변 생성 시작...")
            return self._generate_rag_response(data['query'])
            
        else:
            return "죄송합니다. 질문을 이해하지 못했습니다."

# === 대화형 Agent 실행 ===
if __name__ == '__main__':
    try:
        agent = FigAgent()
        print("\n--- 무화과 품종 챗봇 ---")
        print("안녕하세요! 무화과에 대해 궁금한 점을 물어보세요.")
        print("(종료하시려면 'exit' 또는 'quit'을 입력하세요)")

        while True:
            user_query = input("\n👤 나: ")
            if user_query.lower() in ["exit", "quit"]:
                print("🤖 Agent: 이용해주셔서 감사합니다.")
                break
            
            response = agent.handle_query(user_query)
            print(f"🤖 Agent: {response}")

    except Exception as e:
        print(f"\n❗️ Agent 실행 중 오류 발생: {e}")

import os
import google.generativeai as genai
from dotenv import load_dotenv

# .env 파일에서 API 키를 로드합니다.
load_dotenv()

try:
    # 환경 변수에서 API 키를 가져옵니다.
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 환경 변수를 찾을 수 없습니다. .env 파일을 확인해주세요.")

    # API 키를 사용하여 genai 라이브러리를 설정합니다.
    genai.configure(api_key=api_key)

    print("✅ API 키가 성공적으로 설정되었습니다.")
    print("사용 가능한 모델 목록을 조회합니다...\n")

    # 텍스트 생성(generateContent)을 지원하는 모델만 필터링하여 출력합니다.
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"- 모델 이름: {model.name}")
            print(f"  - 설명: {model.description}\n")

except Exception as e:
    print(f"❌ 오류가 발생했습니다: {e}")
    print("API 키가 올바른지, .env 파일이 정확한지 다시 확인해주세요.")


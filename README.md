# FigVarietyRAGChat

RAG(검색 증강 생성) 기술을 기반으로, 희귀 무화과 품종에 대한 정보를 제공하는 챗봇 프로젝트입니다.

## 📖 프로젝트 구조

- **`fig_agent.py`**: RAG 파이프라인과 사용자 질문 분류 등 챗봇의 핵심 로직이 담긴 `FigAgent` 클래스를 정의합니다.
- **`fig_client.py`**: 사용자와 직접 상호작용하는 대화형 터미널 클라이언트입니다. 챗봇을 실행하려면 이 파일을 사용합니다.
- **`agent_test.py`**: `FigAgent`의 기능을 테스트하기 위한 스크립트입니다. 미리 정의된 질문 목록을 자동으로 실행합니다.
- **`build_features_db.py`**: `features` 디렉토리의 텍스트 파일을 기반으로 특징 벡터 DB(`features_db`)를 생성합니다.
- **`build_varieties_db.py`**: `varieties` 디렉토리의 파일 목록을 기반으로 품종명 벡터 DB(`vector_db`)를 생성합니다.
- **`check_models.py`**: 설정된 API 키로 사용 가능한 Google Gemini 모델 목록을 확인하는 유틸리티 스크립트입니다.
- **`requirements.txt`**: 프로젝트 실행에 필요한 파이썬 라이브러리 목록입니다.
- **`.env`**: API 키 등 민감한 정보를 저장하는 환경 변수 파일입니다. (버전 관리에서 제외됨)

- **`features/`**: 품종별 특징(맛, 생산성 등)에 대한 원본 데이터가 들어있는 디렉토리입니다.
- **`varieties/`**: 품종 목록을 관리하는 디렉토리입니다.
- **`features_db/`**: 특징 정보가 임베딩되어 저장되는 벡터 데이터베이스입니다.
- **`vector_db/`**: 품종명 정보가 임베딩되어 저장되는 벡터 데이터베이스입니다.

## 🚀 실행 방법

### 1. 환경 설정

**가. 라이브러리 설치**
프로젝트에 필요한 라이브러리들을 설치합니다.
```bash
pip install -r requirements.txt
```

**나. API 키 설정**
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고, 그 안에 Google Gemini API 키를 입력합니다.
```
# .env 파일 내용
GEMINI_API_KEY="여기에_실제_API_키를_입력하세요"
```

### 2. 벡터 데이터베이스 생성

처음 실행하거나 `features` 또는 `varieties` 디렉토리의 내용이 변경된 경우, 아래 명령어를 실행하여 벡터 데이터베이스를 다시 생성해야 합니다.

```bash
python build_features_db.py
python build_varieties_db.py
```

### 3. 챗봇 실행

아래 명령어를 실행하여 터미널에서 대화형 챗봇을 시작합니다.

```bash
python fig_client.py
```

### 4. (선택) 에이전트 테스트

미리 정의된 질문들로 에이전트의 응답을 빠르게 확인하고 싶다면 아래 명령어를 실행하세요.

```bash
python agent_test.py
```
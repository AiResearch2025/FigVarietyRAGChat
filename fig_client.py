# === FigAgent 대화형 클라이언트 ===

from fig_agent import FigAgent

def start_chat():
    """
    FigAgent를 사용하여 대화형 챗봇 세션을 시작합니다.
    """
    try:
        # fig_agent.py에 정의된 FigAgent 클래스를 초기화합니다.
        agent = FigAgent()
    except Exception as e:
        print(f"❗️ Agent를 초기화하는 중 오류가 발생했습니다: {e}")
        return

    print("\n--- 무화과 품종 챗봇 ---")
    print("안녕하세요! 무화과에 대해 궁금한 점을 물어보세요.")
    print("(종료하시려면 'exit' 또는 'quit'을 입력하세요)")

    # 사용자와 계속 대화하기 위한 무한 루프
    while True:
        try:
            # 사용자로부터 질문을 입력받습니다.
            user_query = input("\n👤 나: ")

            # 종료 조건 확인
            if user_query.lower() in ["exit", "quit"]:
                print("🤖 Agent: 이용해주셔서 감사합니다.")
                break
            
            # Agent를 통해 응답을 생성하고 출력합니다.
            response = agent.handle_query(user_query)
            print(f"🤖 Agent: {response}")

        except KeyboardInterrupt:
            # Ctrl+C 입력 시 종료
            print("\n🤖 Agent: 채팅을 종료합니다.")
            break
        except Exception as e:
            print(f"❗️ 질문을 처리하는 중 오류가 발생했습니다: {e}")

if __name__ == '__main__':
    start_chat()

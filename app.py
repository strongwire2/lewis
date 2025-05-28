from flask import Flask, render_template, request, Response
from networkx.drawing.nx_pydot import to_pydot

# Flask 애플리케이션 객체 생성
app = Flask(__name__)

# 애플리케이션 설정 (선택 사항)
app.config['SECRET_KEY'] = 'your_very_secret_key_here'  # 폼 CSRF 보호 등에 사용


# 루트 URL ("/")에 대한 핸들러
@app.route('/')
def index():
    """메인 페이지를 보여줍니다."""
    # templates 폴더의 index.html 파일을 렌더링하여 반환
    # messages 리스트를 템플릿에 전달
    return render_template('index.html')


@app.route('/lewis')
def lewis():
    """
    루이스 구조를 구하고 리턴합니다. 일단은 dot 포맷
    :return:
    """
    formula = request.args.get('formula')
    print(f"lewis request for {formula}")

    # TODO: 여기서 버전을 바꿀 수 있음. lewis 대신 lewis_fix 등.
    from lewis import get_lewis_struct, annotate_lewis

    result = get_lewis_struct(formula)

    dot_string = ''
    # TODO: 일단 첫번째 거만 취하자.
    if len(result) > 0:
        annotate_lewis(result[0])
        dot = to_pydot(result[0])
        dot_string = dot.to_string()

    return Response(dot_string, mimetype='text/plain')


# --- 애플리케이션 실행 ---
if __name__ == '__main__':
    # 개발 서버 실행
    # debug=True: 코드 변경 시 자동 재시작, 디버그 정보 제공 (프로덕션에서는 False)
    # host='0.0.0.0': 외부에서도 접속 가능하게 함 (로컬에서만 테스트 시 생략 가능)
    # port=5000: 사용할 포트 번호 (기본값)
    # Web Browser에서 http://localhost:5000 으로 접속
    app.run(debug=True, host='0.0.0.0', port=5000)

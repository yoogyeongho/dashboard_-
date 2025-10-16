import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="가맹점 위기경보 대시보드",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 여러 페이지 정보를 등록
pages = [
    st.Page("마지막 합본.py", title="성동구 가맹점 현황 대시보드", icon="🗺",default=True),
    st.Page("최최종 대시보드 구성.py", title="지역별 위기경보/매출등급", icon="📊"),
    st.Page("test.py", title="업종별 위기경보 등급 변화", icon="📈")

]

# 사용자가 선택한 페이지를 받아오기
selected_page = st.navigation(pages)

# 선택된 페이지 실행
selected_page.run()
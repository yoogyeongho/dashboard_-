import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# CSS 스타일 (다크 테마)
st.markdown("""
    <style>
    .main { 
        background-color: #2b2d42; 
    }
    .stApp {
        background-color: #2b2d42;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: white;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #a0a0a0;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #1a1b2e;
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    div[data-testid="stHorizontalBlock"] {
        background-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

# XGBoost 모델 학습 함수
@st.cache_resource
def train_xgboost_model(df_full):
    """XGBoost 모델 학습"""
    
    df_model = df_full.copy()
    
    if '기준년월' in df_model.columns:
        df_model['기준년월'] = pd.to_datetime(df_model['기준년월'], format='%Y%m', errors='coerce')
        df_model['기준년'] = df_model['기준년월'].dt.year.fillna(0).astype(int)
        df_model['기준월'] = df_model['기준년월'].dt.month.fillna(0).astype(int)
        df_model['month_sin'] = np.sin(2 * np.pi * (df_model['기준월'] - 1) / 12).astype('float32')
        df_model['month_cos'] = np.cos(2 * np.pi * (df_model['기준월'] - 1) / 12).astype('float32')
    
    df_model = df_model.drop(columns=['기준년월', '기준월'], errors='ignore')
    
    excluded_cols = [
        '기준년', '폐업일', '가맹점주소', '가맹점명', '가맹점지역', '객단가 구간', 
        '가맹점구분번호', '개설일', '경도', '위도', '임대료 점수', '소비액 점수', 
        '가맹점 이용 직장인구 수', '가맹점 이용 상주인구 수', '가맹점 이용 유동인구 수',
        '년월', '년월_str'
    ]
    
    features = [col for col in df_model.columns if col not in excluded_cols and col != '폐업여부']
    X_all = df_model[features].fillna(0)
    y_all = df_model['폐업여부']
    
    le_dict = {}
    for col in ['경쟁과열', '행정동', '업종']:
        if col in X_all.columns and X_all[col].dtype == 'object':
            le = LabelEncoder()
            X_all[col] = le.fit_transform(X_all[col].astype(str))
            le_dict[col] = le
    
    train_mask = df_model['기준년'] == 2023
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1
    
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    return model, features, le_dict

# 폐업 확률 예측 함수
def predict_closure_probability(df, model, features, le_dict):
    """폐업 확률 예측"""
    
    df_pred = df.copy()
    
    if '기준년월' in df_pred.columns:
        df_pred['기준년월_dt'] = pd.to_datetime(df_pred['기준년월'].astype(str), format='%Y%m', errors='coerce')
        df_pred['기준년'] = df_pred['기준년월_dt'].dt.year.fillna(0).astype(int)
        df_pred['기준월'] = df_pred['기준년월_dt'].dt.month.fillna(0).astype(int)
        df_pred['month_sin'] = np.sin(2 * np.pi * (df_pred['기준월'] - 1) / 12).astype('float32')
        df_pred['month_cos'] = np.cos(2 * np.pi * (df_pred['기준월'] - 1) / 12).astype('float32')
    
    X_pred = pd.DataFrame(index=df_pred.index)
    for feature in features:
        if feature in df_pred.columns:
            X_pred[feature] = df_pred[feature].fillna(0)
        else:
            X_pred[feature] = 0
    
    for col, le in le_dict.items():
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].apply(
                lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0
            )
    
    probabilities = model.predict_proba(X_pred)[:, 1]
    
    return probabilities

# 위기경보 등급 계산 함수
def calculate_warning_grade(df, model, features, le_dict):
    """폐업 확률 기반 위기경보 등급 계산"""
    
    closure_probs = predict_closure_probability(df, model, features, le_dict)
    df['폐업확률'] = closure_probs
    
    absolute_thresholds = {
        '위험': 0.6,
        '주의': 0.4,
        '관심': 0.2,
        '안정': 0.0
    }
    
    percentiles = df['폐업확률'].quantile([0.75, 0.5, 0.25]).values
    
    weight_absolute = 0.6
    weight_relative = 0.4
    
    hybrid_thresholds = [
        weight_absolute * absolute_thresholds['위험'] + weight_relative * percentiles[0],
        weight_absolute * absolute_thresholds['주의'] + weight_relative * percentiles[1],
        weight_absolute * absolute_thresholds['관심'] + weight_relative * percentiles[2]
    ]
    
    conditions = [
        df['폐업확률'] >= hybrid_thresholds[0],
        (df['폐업확률'] >= hybrid_thresholds[1]) & (df['폐업확률'] < hybrid_thresholds[0]),
        (df['폐업확률'] >= hybrid_thresholds[2]) & (df['폐업확률'] < hybrid_thresholds[1]),
        df['폐업확률'] < hybrid_thresholds[2]
    ]
    choices = ['위험', '주의', '관심', '안정']
    
    df['조기경보등급'] = np.select(conditions, choices, default='안정')
    
    return df

# 데이터 로드 및 전처리
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("final_df.zip")
    
    # 기준년월을 datetime으로 변환
    df['년월'] = pd.to_datetime(df['기준년월'].astype(str), format='%Y%m')
    df['년월_str'] = df['년월'].dt.strftime('%Y년 %m월')
    
    return df

# 순위별 바 차트 생성 함수
def create_ranking_bar_chart(data_df, value_col, name_col, title, color_scale, limit=20):
    """순위별 바 차트 생성"""
    top_data = data_df.head(limit)
    
    top_data['순위'] = range(1, len(top_data) + 1)
    
    colors = []
    for i in range(len(top_data)):
        if i < 3:
            colors.append(f'rgb({255-i*30}, {50+i*20}, {50+i*20})')
        elif i < 10:
            colors.append(f'rgb({150-i*5}, {50+i*10}, {150+i*5})')
        else:
            colors.append(f'rgb({50+i*3}, {50+i*3}, {200-i*5})')
    
    fig = go.Figure()
    
    for idx, row in top_data.iterrows():
        fig.add_trace(go.Bar(
            x=[row[value_col]],
            y=[row['순위']],
            orientation='h',
            name='',
            showlegend=False,
            marker_color=colors[row['순위']-1],
            text=f"{row[name_col]} {row[value_col]:.1f}%",
            textposition='outside',
            textfont=dict(color='white', size=11),
            hovertemplate=f"<b>{row[name_col]}</b><br>{row[value_col]:.1f}%<extra></extra>"
        ))
    
    fig.update_layout(
        title=title,
        title_font=dict(size=14, color='white'),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            range=[0, top_data[value_col].max() * 1.3]
        ),
        yaxis=dict(
            showgrid=False,
            autorange='reversed',
            tickmode='array',
            tickvals=top_data['순위'].tolist(),
            ticktext=[f"{i}위" for i in top_data['순위']],
            tickfont=dict(color='white', size=10)
        ),
        height=500,
        margin=dict(l=50, r=150, t=40, b=30),
        plot_bgcolor='#2b2d42',
        paper_bgcolor='#2b2d42',
        bargap=0.3,
        font=dict(color='white')
    )
    
    return fig

# 등급별 바 차트 생성
def create_grade_bar_chart(grade_data, selected_grade=None):
    """조기경보 등급별 바 차트"""
    colors = {
        '안정': '#00ff88',
        '관심': '#ffcc00', 
        '주의': '#ff8800',
        '위험': '#ff4444'
    }
    
    fig = go.Figure()
    
    for idx, (grade, ratio) in enumerate(grade_data.items()):
        opacity = 1.0 if selected_grade is None or selected_grade == grade else 0.3
        
        fig.add_trace(go.Bar(
            x=[ratio],
            y=[grade],
            orientation='h',
            marker_color=colors[grade],
            opacity=opacity,
            name='',
            showlegend=False,
            text=f"{ratio:.1f}%",
            textposition='outside',
            textfont=dict(color='white', size=11),
            hovertemplate=f"<b>{grade}</b><br>{ratio:.1f}%<extra></extra>"
        ))
    
    fig.update_layout(
        title="조기경보 등급 분포",
        title_font=dict(size=14, color='white'),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            range=[0, max(grade_data.values()) * 1.3]
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color='white', size=11)
        ),
        height=500,
        margin=dict(l=60, r=100, t=40, b=30),
        plot_bgcolor='#2b2d42',
        paper_bgcolor='#2b2d42',
        bargap=0.4,
        font=dict(color='white')
    )
    
    return fig

def main():
    # 제목
    st.markdown('<h1 class="main-header">업종별 위기경보 등급 변화</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">위기경보등급 위험 업종 변화 (2023.01~2024.12)</p>', unsafe_allow_html=True)
    
    # 데이터 로드
    df = load_and_preprocess_data()
    
    # XGBoost 모델 학습
    try:
        model, features, le_dict = train_xgboost_model(df)
    except Exception as e:
        st.error(f"모델 학습 중 오류 발생: {str(e)}")
        return
    
    # 세션 상태 초기화
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
    if 'current_date_idx' not in st.session_state:
        st.session_state.current_date_idx = 0
    if 'selected_dong' not in st.session_state:
        st.session_state.selected_dong = None
    if 'selected_grade' not in st.session_state:
        st.session_state.selected_grade = None
    if 'selected_business' not in st.session_state:
        st.session_state.selected_business = None
    
    # 날짜 리스트 생성
    dates = sorted(df['년월_str'].unique())
    
    # 컨트롤 패널
    control_row = st.container()
    with control_row:
        left_col, spacer_col, right_col = st.columns([2, 6, 10])
        
        # 왼쪽: 재생/초기화 버튼 그룹
        with left_col:
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                play_button = st.button("▶ 재생" if not st.session_state.is_playing else "⏸ 일시정지", use_container_width=True)
                if play_button:
                    st.session_state.is_playing = not st.session_state.is_playing
            
            with btn_col2:
                reset_button = st.button("🔄 초기화", use_container_width=True)
                if reset_button:
                    st.session_state.current_date_idx = 0
                    st.session_state.selected_dong = None
                    st.session_state.selected_grade = None
                    st.session_state.selected_business = None
                    st.session_state.is_playing = False
                    st.rerun()
        
        # 오른쪽: 날짜 조정 그룹
        with right_col:
            date_col1, date_col2, date_col3 = st.columns([0.5, 9, 0.5])
            
            with date_col1:
                if st.button("◀", use_container_width=True):
                    if st.session_state.current_date_idx > 0:
                        st.session_state.current_date_idx -= 1
                        st.rerun()
            
            with date_col2:
                selected_date_idx = st.slider(
                    "기준 날짜",
                    min_value=0,
                    max_value=len(dates)-1,
                    value=st.session_state.current_date_idx,
                    format=f"%d",
                    key="date_slider"
                )
                if selected_date_idx != st.session_state.current_date_idx:
                    st.session_state.current_date_idx = selected_date_idx
                    st.rerun()
            
            with date_col3:
                if st.button("▶", use_container_width=True):
                    if st.session_state.current_date_idx < len(dates) - 1:
                        st.session_state.current_date_idx += 1
                        st.rerun()
    
    # 현재 날짜 표시
    current_date = dates[st.session_state.current_date_idx]
    st.markdown(f"<h3 style='text-align:center; color:white;'>📅 {current_date}</h3>", unsafe_allow_html=True)
    
    # 현재 날짜 데이터 필터링 및 등급 계산
    current_df = df[df['년월_str'] == current_date].copy()
    current_df = calculate_warning_grade(current_df, model, features, le_dict)
    
    # 메인 차트 컨테이너
    chart_placeholder = st.empty()
    
    with chart_placeholder.container():
        col1, col2, col3, col4 = st.columns([1.2, 0.8, 1.2, 1.2])
        
        # 1. 행정동별 위험 비율 순위
        with col1:
            # 필터 적용
            if st.session_state.selected_dong:
                dong_df = current_df[current_df['행정동'] == st.session_state.selected_dong]
            else:
                dong_df = current_df
            
            # 행정동별 위험 비율 계산
            dong_danger = []
            for dong in dong_df['행정동'].unique():
                dong_data = dong_df[dong_df['행정동'] == dong]
                total_count = len(dong_data)
                danger_count = len(dong_data[dong_data['조기경보등급'] == '위험'])
                if total_count > 0:
                    danger_ratio = (danger_count / total_count) * 100
                    dong_danger.append({
                        '행정동': dong,
                        '위험비율': danger_ratio,
                        '위험가맹점수': danger_count
                    })
            
            if dong_danger:
                dong_result = pd.DataFrame(dong_danger).sort_values('위험비율', ascending=False)
                fig1 = create_ranking_bar_chart(
                    dong_result, 
                    '위험비율', 
                    '행정동',
                    "행정동별 위험 비율 순위",
                    'Reds'
                )
                st.plotly_chart(fig1, use_container_width=True, key=f"dong_chart_{current_date}")
                
                # 클릭 이벤트 처리 (선택 UI)
                if not st.session_state.selected_dong:
                    selected_dong_input = st.selectbox(
                        "행정동 선택",
                        options=[None] + dong_result['행정동'].tolist(),
                        index=0,
                        key=f"dong_select_{current_date}"
                    )
                    if selected_dong_input and selected_dong_input != st.session_state.selected_dong:
                        st.session_state.selected_dong = selected_dong_input
                        st.rerun()
                else:
                    st.info(f"선택된 행정동: {st.session_state.selected_dong}")
                    if st.button("행정동 선택 해제", key=f"clear_dong_{current_date}"):
                        st.session_state.selected_dong = None
                        st.session_state.selected_grade = None
                        st.session_state.selected_business = None
                        st.rerun()
        
        # 2. 조기경보 등급별 비율
        with col2:
            # 선택된 행정동 필터 적용
            if st.session_state.selected_dong:
                grade_df = current_df[current_df['행정동'] == st.session_state.selected_dong]
            else:
                grade_df = current_df
            
            # 등급별 비율 계산
            total = len(grade_df)
            grade_ratios = {}
            for grade in ['위험', '주의', '관심', '안정']:
                count = len(grade_df[grade_df['조기경보등급'] == grade])
                grade_ratios[grade] = (count / total * 100) if total > 0 else 0
            
            fig2 = create_grade_bar_chart(grade_ratios, st.session_state.selected_grade)
            st.plotly_chart(fig2, use_container_width=True, key=f"grade_chart_{current_date}")
            
            # 등급 선택
            if st.session_state.selected_dong and not st.session_state.selected_grade:
                selected_grade_input = st.selectbox(
                    "등급 선택",
                    options=[None, '위험', '주의', '관심', '안정'],
                    index=0,
                    key=f"grade_select_{current_date}"
                )
                if selected_grade_input and selected_grade_input != st.session_state.selected_grade:
                    st.session_state.selected_grade = selected_grade_input
                    st.rerun()
            elif st.session_state.selected_grade:
                st.info(f"선택된 등급: {st.session_state.selected_grade}")
                if st.button("등급 선택 해제", key=f"clear_grade_{current_date}"):
                    st.session_state.selected_grade = None
                    st.session_state.selected_business = None
                    st.rerun()
        
        # 3. 업종별 위험 비율 순위
        with col3:
            # 필터 적용
            if st.session_state.selected_dong and st.session_state.selected_grade:
                business_df = current_df[
                    (current_df['행정동'] == st.session_state.selected_dong) &
                    (current_df['조기경보등급'] == st.session_state.selected_grade)
                ]
                # 해당 행정동의 업종별 위험 비율
                business_danger = []
                dong_df = current_df[current_df['행정동'] == st.session_state.selected_dong]
                for business in dong_df['업종'].unique():
                    business_data = dong_df[dong_df['업종'] == business]
                    total_count = len(business_data)
                    danger_count = len(business_data[business_data['조기경보등급'] == st.session_state.selected_grade])
                    if total_count > 0:
                        danger_ratio = (danger_count / total_count) * 100
                        business_danger.append({
                            '업종': business,
                            '위험비율': danger_ratio,
                            '가맹점수': danger_count
                        })
            else:
                # 전체 업종별 위험 비율
                business_danger = []
                for business in current_df['업종'].unique():
                    business_data = current_df[current_df['업종'] == business]
                    total_count = len(business_data)
                    danger_count = len(business_data[business_data['조기경보등급'] == '위험'])
                    if total_count > 0:
                        danger_ratio = (danger_count / total_count) * 100
                        business_danger.append({
                            '업종': business,
                            '위험비율': danger_ratio,
                            '가맹점수': danger_count
                        })
            
            if business_danger:
                business_result = pd.DataFrame(business_danger).sort_values('위험비율', ascending=False)
                fig3 = create_ranking_bar_chart(
                    business_result,
                    '위험비율',
                    '업종',
                    "업종별 위험 비율 순위",
                    'Purples'
                )
                st.plotly_chart(fig3, use_container_width=True, key=f"business_chart_{current_date}")
                
                # 업종 선택
                if st.session_state.selected_dong and st.session_state.selected_grade and not st.session_state.selected_business:
                    selected_business_input = st.selectbox(
                        "업종 선택",
                        options=[None] + business_result['업종'].tolist(),
                        index=0,
                        key=f"business_select_{current_date}"
                    )
                    if selected_business_input and selected_business_input != st.session_state.selected_business:
                        st.session_state.selected_business = selected_business_input
                        st.rerun()
                elif st.session_state.selected_business:
                    st.info(f"선택된 업종: {st.session_state.selected_business}")
                    if st.button("업종 선택 해제", key=f"clear_business_{current_date}"):
                        st.session_state.selected_business = None
                        st.rerun()
        
        # 4. 가맹점별 위험 순위
        with col4:
            # 필터 적용
            if st.session_state.selected_dong and st.session_state.selected_grade and st.session_state.selected_business:
                store_df = current_df[
                    (current_df['행정동'] == st.session_state.selected_dong) &
                    (current_df['조기경보등급'] == st.session_state.selected_grade) &
                    (current_df['업종'] == st.session_state.selected_business)
                ]
            elif st.session_state.selected_dong and st.session_state.selected_grade:
                store_df = current_df[
                    (current_df['행정동'] == st.session_state.selected_dong) &
                    (current_df['조기경보등급'] == st.session_state.selected_grade)
                ]
            else:
                # 전체 위험 가맹점
                store_df = current_df[current_df['조기경보등급'] == '위험']
            
            # 가맹점별 점수로 정렬 (폐업확률이 높을수록 위험)
            if not store_df.empty:
                store_ranking = store_df.sort_values('폐업확률', ascending=False).head(20)
                store_danger = []
                
                for _, row in store_ranking.iterrows():
                    store_danger.append({
                        '가맹점': row['가맹점구분번호'][:8] + "...",
                        '위험점수': row['폐업확률'] * 100,  # 확률을 점수로 변환
                        '업종': row['업종']
                    })
                
                if store_danger:
                    store_result = pd.DataFrame(store_danger)
                    fig4 = create_ranking_bar_chart(
                        store_result,
                        '위험점수',
                        '가맹점',
                        "가맹점별 위험 순위",
                        'Blues'
                    )
                    st.plotly_chart(fig4, use_container_width=True, key=f"store_chart_{current_date}")
            else:
                st.info("선택 조건에 해당하는 가맹점이 없습니다.")
    
    # 자동 재생 기능
    if st.session_state.is_playing:
        time.sleep(1)  # 1초 대기
        
        # 다음 날짜로 이동
        st.session_state.current_date_idx += 1
        if st.session_state.current_date_idx >= len(dates):
            st.session_state.current_date_idx = 0  # 처음으로 돌아가기
            st.session_state.is_playing = False  # 재생 중지
        
        st.rerun()
    
    # 선택 정보 표시
    if st.session_state.selected_dong or st.session_state.selected_grade or st.session_state.selected_business:
        st.markdown("---")
        selected_info = []
        if st.session_state.selected_dong:
            selected_info.append(f"행정동: {st.session_state.selected_dong}")
        if st.session_state.selected_grade:
            selected_info.append(f"등급: {st.session_state.selected_grade}")
        if st.session_state.selected_business:
            selected_info.append(f"업종: {st.session_state.selected_business}")
        
        st.markdown(f"<p style='color: #a0a0a0;'>🔍 선택된 필터: {' > '.join(selected_info)}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
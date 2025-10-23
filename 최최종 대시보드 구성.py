import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# CSS 스타일 - 흰색 배경 버전
st.markdown("""
    <style>
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    .danger { color: #d32f2f; }
    .warning { color: #f57c00; }
    .caution { color: #fbc02d; }
    .safe { color: #388e3c; }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1a1a1a;
    }
    </style>
""", unsafe_allow_html=True)

# XGBoost 모델 학습 함수
@st.cache_resource
def train_xgboost_model(df_full):
    """final_df 데이터로 XGBoost 모델 학습"""
    
    df_model = df_full.copy()
    
    # 기준년월 처리 및 푸리에 변환
    if '기준년월' in df_model.columns:
        df_model['기준년월'] = pd.to_datetime(df_model['기준년월'], format='%Y%m', errors='coerce')
        df_model['기준년'] = df_model['기준년월'].dt.year.fillna(0).astype(int)
        df_model['기준월'] = df_model['기준년월'].dt.month.fillna(0).astype(int)
        df_model['month_sin'] = np.sin(2 * np.pi * (df_model['기준월'] - 1) / 12).astype('float32')
        df_model['month_cos'] = np.cos(2 * np.pi * (df_model['기준월'] - 1) / 12).astype('float32')
    
    df_model = df_model.drop(columns=['기준년월', '기준월'], errors='ignore')
    
    # Feature 선택
    excluded_cols = [
        '기준년', '폐업일', '가맹점주소', '가맹점명', '가맹점지역', '객단가 구간', 
        '가맹점구분번호', '개설일', '경도', '위도', '임대료 점수', '소비액 점수', 
        '가맹점 이용 직장인구 수', '가맹점 이용 상주인구 수', '가맹점 이용 유동인구 수',
        '년월', '년월_str'
    ]
    
    features = [col for col in df_model.columns if col not in excluded_cols and col != '폐업여부']
    X_all = df_model[features].fillna(0)
    y_all = df_model['폐업여부']
    
    # 범주형 변수 인코딩
    le_dict = {}
    for col in ['경쟁과열', '행정동', '업종']:
        if col in X_all.columns and X_all[col].dtype == 'object':
            le = LabelEncoder()
            X_all[col] = le.fit_transform(X_all[col].astype(str))
            le_dict[col] = le
    
    # 2023년 데이터로만 학습
    train_mask = df_model['기준년'] == 2023
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    
    # scale_pos_weight 계산
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1
    
    # 모델 학습
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    
    model.fit(X_train, y_train)
    
    return model, features, le_dict

# 폐업 확률 예측 함수
@st.cache_data
def predict_closure_probability(df, _model, features, _le_dict):
    """폐업 확률 예측 (캐싱)"""
    
    df_pred = df.copy()
    
    # 기준년월 처리 및 푸리에 변환
    if '기준년월' in df_pred.columns:
        df_pred['기준년월_dt'] = pd.to_datetime(df_pred['기준년월'].astype(str), format='%Y%m', errors='coerce')
        df_pred['기준년'] = df_pred['기준년월_dt'].dt.year.fillna(0).astype(int)
        df_pred['기준월'] = df_pred['기준년월_dt'].dt.month.fillna(0).astype(int)
        df_pred['month_sin'] = np.sin(2 * np.pi * (df_pred['기준월'] - 1) / 12).astype('float32')
        df_pred['month_cos'] = np.cos(2 * np.pi * (df_pred['기준월'] - 1) / 12).astype('float32')
    
    # Feature 준비
    X_pred = pd.DataFrame(index=df_pred.index)
    for feature in features:
        if feature in df_pred.columns:
            X_pred[feature] = df_pred[feature].fillna(0)
        else:
            X_pred[feature] = 0
    
    # 범주형 변수 인코딩
    for col, le in _le_dict.items():
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].apply(
                lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0
            )
    
    # 예측 (_model 사용 - 언더스코어는 캐싱 무시)
    probabilities = _model.predict_proba(X_pred)[:, 1]
    
    return probabilities

# 위기경보 등급 계산 함수
@st.cache_data
def calculate_warning_grade(df, _model=None, features=None, _le_dict=None, _model_available=True):
    """폐업 확률 기반 위기경보 등급 계산 (캐싱)"""
    
    df = df.copy()  # 원본 데이터프레임 보호
    
    if _model_available and _model is not None:
        # 폐업 확률 예측
        closure_probs = predict_closure_probability(df, _model, features, _le_dict)
        df['폐업확률'] = closure_probs
        
        # 하이브리드 방식으로 등급 분류
        absolute_thresholds = {
            '위험': 0.52,
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
        
        df['위기경보등급'] = np.select(conditions, choices, default='안정')
            
    else:
        # Fallback
        percentiles = df['회복탄력성_점수'].quantile([0.25, 0.5, 0.75]).tolist()
        df['위기경보등급'] = pd.cut(
            df['회복탄력성_점수'],
            bins=[-np.inf] + percentiles + [np.inf],
            labels=['위험', '주의', '관심', '안정']
        )
    
    return df

# 데이터 로드 함수
@st.cache_data
def load_data():
    """final_df.csv에서 모든 데이터 로드"""
    try:
        df = pd.read_csv("final_df.zip")
    except FileNotFoundError:
        st.error("final_df.zip 파일을 찾을 수 없습니다.")
        return pd.DataFrame()
    
    # 기준년월을 datetime으로 변환
    if '기준년월' in df.columns:
        df['년월'] = pd.to_datetime(df['기준년월'].astype(str), format='%Y%m', errors='coerce')
        df['년월_str'] = df['년월'].dt.strftime('%Y년 %m월')
    
    return df

# 생존 분석 데이터 준비 함수
@st.cache_data
def prepare_survival_data(df):
    """생존 분석을 위한 데이터 준비 (캐싱)"""
    
    if '가맹점구분번호' in df.columns:
        franchise_id = '가맹점구분번호'
    elif '가맹점명' in df.columns:
        franchise_id = '가맹점명'
    else:
        return pd.DataFrame()
    
    df_survival = df.copy()
    
    df_survival['개설일_dt'] = pd.to_datetime(df_survival['개설일'], format='%Y%m%d', errors='coerce')
    
    # 개설일이 유효하지 않거나 1900년 이전인 경우 제외
    df_survival = df_survival[
        (df_survival['개설일_dt'].notna()) & 
        (df_survival['개설일_dt'] >= '1900-01-01') &
        (df_survival['개설일_dt'] <= '2024-12-31')
    ].copy()
    
    if '폐업일' in df_survival.columns:
        df_survival['폐업일_dt'] = pd.to_datetime(df_survival['폐업일'], format='%Y%m%d', errors='coerce')
    else:
        df_survival['폐업일_dt'] = pd.NaT
    
    if '폐업여부' in df_survival.columns:
        if df_survival['폐업여부'].dtype == 'object':
            df_survival['폐업여부'] = df_survival['폐업여부'].map(
                {'Y': 1, 'N': 0, '1': 1, '0': 0, 1: 1, 0: 0}
            ).fillna(0).astype(int)
    else:
        df_survival['폐업여부'] = df_survival['폐업일_dt'].notna().astype(int)
    
    # 분석 기준일을 2024년 12월 31일로 고정
    end_date = pd.to_datetime('20241231', format='%Y%m%d')
    
    unique_franchises = []
    
    for fid in df_survival[franchise_id].unique():
        franchise_records = df_survival[df_survival[franchise_id] == fid].sort_values('기준년월')
        
        if franchise_records.empty:
            continue
        
        first_rec = franchise_records.iloc[0]
        개설일 = first_rec['개설일_dt']
        행정동 = first_rec.get('행정동', None)
        업종 = first_rec.get('업종', None)
        
        last_rec = franchise_records.iloc[-1]
        폐업여부 = last_rec['폐업여부']
        폐업일 = last_rec['폐업일_dt']
        
        # 폐업한 경우 폐업일, 아니면 고정 기준일 사용
        if pd.notna(폐업일) and 폐업여부 == 1:
            종료일 = 폐업일
        else:
            종료일 = end_date
        
        if pd.notna(개설일) and pd.notna(종료일):
            운영기간_개월 = (종료일 - 개설일).days / 30.0
            
            # 운영기간이 0~600개월(50년) 범위 내인 경우만 포함
            if 0 <= 운영기간_개월 <= 600:
                unique_franchises.append({
                    franchise_id: fid,
                    '행정동': 행정동,
                    '업종': 업종,
                    '운영기간_개월': 운영기간_개월,
                    '폐업여부': 폐업여부
                })
    
    unique_franchises_df = pd.DataFrame(unique_franchises)
    
    if franchise_id in unique_franchises_df.columns:
        unique_franchises_df = unique_franchises_df.drop_duplicates(subset=[franchise_id]).copy()
    
    # 비정상적인 운영기간 데이터 제거 (추가 검증)
    if not unique_franchises_df.empty and '운영기간_개월' in unique_franchises_df.columns:
        unique_franchises_df = unique_franchises_df[
            (unique_franchises_df['운영기간_개월'] >= 0) & 
            (unique_franchises_df['운영기간_개월'] <= 600)
        ].copy()
    
    return unique_franchises_df

# 메인 함수
def main():
    # 제목
    st.markdown('<h1 class="main-header">지역별 위기경보 / 매출 등급 대시보드</h1>', unsafe_allow_html=True)
    
    # 데이터 로드
    df = load_data()
    
    if df.empty:
        return
    
    # XGBoost 모델 학습
    try:
        model, features, le_dict = train_xgboost_model(df)
        model_available = True
    except Exception as e:
        st.warning(f"모델 학습 중 오류 발생: {str(e)}. 기존 방식으로 위기경보 등급을 계산합니다.")
        model = None
        features = None
        le_dict = None
        model_available = False
    
    # 필터 컨테이너
    filter_container = st.container()
    with filter_container:
        col1, col2, col3 = st.columns(3)
        
        # 행정동 필터
        with col1:
            selected_dong = st.selectbox(
                "행정동명",
                options=['전체'] + sorted(df['행정동'].unique().tolist()),
                index=0
            )
        
        # 행정동 필터링
        if selected_dong != '전체':
            filtered_df = df[df['행정동'] == selected_dong]
        else:
            filtered_df = df.copy()
        
        # 업종 필터
        with col2:
            available_businesses = sorted(filtered_df['업종'].unique().tolist())
            selected_business = st.selectbox(
                "업종명",
                options=['전체'] + available_businesses,
                index=0
            )
        
        # 기준 날짜 필터
        with col3:
            if selected_business != '전체':
                date_filtered_df = filtered_df[filtered_df['업종'] == selected_business]
            else:
                date_filtered_df = filtered_df.copy()
            
            available_dates = sorted(date_filtered_df['년월_str'].unique().tolist())
            
            if available_dates:
                default_index = len(available_dates) - 1
                selected_date = st.selectbox(
                    "기준 날짜",
                    options=available_dates,
                    index=default_index
                )
            else:
                st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
                return
    
    # 선택된 날짜의 데이터 필터링
    current_df = date_filtered_df[date_filtered_df['년월_str'] == selected_date].copy()
    
    # 이전 월 데이터 가져오기
    current_date = pd.to_datetime(selected_date, format='%Y년 %m월')
    prev_date = current_date - pd.DateOffset(months=1)
    prev_date_str = prev_date.strftime('%Y년 %m월')
    
    if prev_date_str in date_filtered_df['년월_str'].values:
        prev_df = date_filtered_df[date_filtered_df['년월_str'] == prev_date_str].copy()
    else:
        prev_df = pd.DataFrame()
    
    # 위기경보 등급 계산
    if model_available:
        current_df = calculate_warning_grade(current_df, _model=model, features=features, _le_dict=le_dict, _model_available=True)
        if not prev_df.empty:
            prev_df = calculate_warning_grade(prev_df, _model=model, features=features, _le_dict=le_dict, _model_available=True)
    else:
        current_df = calculate_warning_grade(current_df, _model_available=False)
        if not prev_df.empty:
            prev_df = calculate_warning_grade(prev_df, _model_available=False)
    
    # 메트릭 컨테이너
    metrics_container = st.container()
    with metrics_container:
        
        col1, col2 = st.columns(2)
        
        # 위기경보등급 표시
        with col1:
            st.markdown("#### 위기경보등급")
            warning_cols = st.columns(4)
            
            grades = ['안정', '관심', '주의', '위험']
            colors = ['safe', 'caution', 'warning', 'danger']
            
            for idx, (grade, color) in enumerate(zip(grades, colors)):
                with warning_cols[idx]:
                    current_count = len(current_df[current_df['위기경보등급'] == grade])
                    
                    if not prev_df.empty:
                        prev_count = len(prev_df[prev_df['위기경보등급'] == grade])
                        change = ((current_count - prev_count) / prev_count * 100) if prev_count > 0 else 0
                        change_str = f"▲{change:.1f}%" if change >= 0 else f"▼{abs(change):.1f}%"
                    else:
                        change_str = "-"
                    
                    st.markdown(f"""
                        <div class="metric-card">
                            <h4 class="{color}">{grade}</h4>
                            <p style="font-size: 1.5rem; margin: 0; color: #1a1a1a;">{current_count}</p>
                            <p style="font-size: 0.9rem; margin: 0; color: #666;">{change_str}</p>
                        </div>
                    """, unsafe_allow_html=True)
        
        # 매출등급 표시
        with col2:
            st.markdown("#### 매출등급")
            sales_cols = st.columns(6)
            
            for grade in range(1, 7):
                with sales_cols[grade-1]:
                    current_count = len(current_df[current_df['매출금액 구간'] == grade])
                    
                    if not prev_df.empty:
                        prev_count = len(prev_df[prev_df['매출금액 구간'] == grade])
                        change = ((current_count - prev_count) / prev_count * 100) if prev_count > 0 else 0
                        change_str = f"▲{change:.1f}%" if change >= 0 else f"▼{abs(change):.1f}%"
                    else:
                        change_str = "-"
                    
                    st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #1a1a1a;">{grade}등급</h4>
                            <p style="font-size: 1.5rem; margin: 0; color: #1a1a1a;">{current_count}</p>
                            <p style="font-size: 0.9rem; margin: 0; color: #666;">{change_str}</p>
                        </div>
                    """, unsafe_allow_html=True)
    
    # 도넛 차트 섹션
    charts_container = st.container()
    with charts_container:
        col1, col2, col3, col4 = st.columns([1.5, 1, 1.5, 1])
        
        # 위기경보 등급별 비율 도넛차트
        with col1:
            st.markdown("#### 위기경보 등급 구성")
            
            warning_counts = current_df['위기경보등급'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=warning_counts.index.tolist(),
                values=warning_counts.values.tolist(),
                hole=.5,
                marker=dict(colors=['#388e3c', '#fbc02d', '#f57c00', '#d32f2f']),
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(
                annotations=[dict(text=selected_dong if selected_dong != '전체' else '전체', 
                                x=0.5, y=0.5, font_size=16, showarrow=False, font_color='#1a1a1a')],
                showlegend=True,
                height=300,
                margin=dict(t=40, b=40, l=40, r=40),
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='#1a1a1a')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 위기경보 등급별 상세 테이블
        with col2:
            st.markdown("#### ")
            
            table_data = []
            for grade in ['안정', '관심', '주의', '위험']:
                current_count = len(current_df[current_df['위기경보등급'] == grade])
                
                if not prev_df.empty:
                    prev_count = len(prev_df[prev_df['위기경보등급'] == grade])
                    change = ((current_count - prev_count) / prev_count * 100) if prev_count > 0 else 0
                    change_str = f"▲{change:.1f}%" if change >= 0 else f"▼{abs(change):.1f}%"
                else:
                    change_str = "-"
                
                table_data.append({
                    '등급': grade,
                    '증감률': change_str,
                    '가맹점수': current_count
                })
            
            df_table = pd.DataFrame(table_data)
            
            html = '<table style="width:100%; color:#1a1a1a; border-collapse: collapse;">'
            html += '<tr style="border-bottom: 2px solid #e0e0e0;"><th style="padding: 8px;">등급</th><th style="padding: 8px;">증감률</th><th style="padding: 8px;">가맹점수</th></tr>'
            for _, row in df_table.iterrows():
                color = 'color:#d32f2f; font-weight: bold;' if row['등급'] == '위험' else ''
                html += f'<tr style="border-bottom: 1px solid #e0e0e0;"><td style="padding: 8px;">{row["등급"]}</td><td style="padding: 8px;">{row["증감률"]}</td><td style="padding: 8px; {color}">{row["가맹점수"]}</td></tr>'
            html += '</table>'
            
            st.markdown(html, unsafe_allow_html=True)
        
        # 매출 등급별 비율 도넛차트
        with col3:
            st.markdown("#### 매출등급 위험도 구성")
            
            sales_counts = current_df['매출금액 구간'].value_counts().sort_index()
            
            colors = ['#388e3c', '#66bb6a', '#fbc02d', '#f57c00', '#ff6f00', '#d32f2f']
            
            fig = go.Figure(data=[go.Pie(
                labels=[f'{i}등급' for i in sales_counts.index],
                values=sales_counts.values.tolist(),
                hole=.5,
                marker=dict(colors=colors[:len(sales_counts)]),
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(
                annotations=[dict(text=selected_dong if selected_dong != '전체' else '전체', 
                                x=0.5, y=0.5, font_size=16, showarrow=False, font_color='#1a1a1a')],
                showlegend=True,
                height=300,
                margin=dict(t=40, b=40, l=40, r=40),
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='#1a1a1a')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 매출 등급별 상세 테이블
        with col4:
            st.markdown("#### ")
            
            table_data = []
            for grade in range(1, 7):
                current_count = len(current_df[current_df['매출금액 구간'] == grade])
                
                if not prev_df.empty:
                    prev_count = len(prev_df[prev_df['매출금액 구간'] == grade])
                    change = ((current_count - prev_count) / prev_count * 100) if prev_count > 0 else 0
                    change_str = f"▲{change:.1f}%" if change >= 0 else f"▼{abs(change):.1f}%"
                else:
                    change_str = "-"
                
                table_data.append({
                    '등급': f'{grade}등급',
                    '증감률': change_str,
                    '가맹점수': current_count
                })
            
            df_table = pd.DataFrame(table_data)
            
            html = '<table style="width:100%; color:#1a1a1a; border-collapse: collapse;">'
            html += '<tr style="border-bottom: 2px solid #e0e0e0;"><th style="padding: 8px;">등급</th><th style="padding: 8px;">증감률</th><th style="padding: 8px;">가맹점수</th></tr>'
            for _, row in df_table.iterrows():
                color = 'color:#d32f2f; font-weight: bold;' if row['등급'] in ['5등급', '6등급'] else ''
                html += f'<tr style="border-bottom: 1px solid #e0e0e0;"><td style="padding: 8px;">{row["등급"]}</td><td style="padding: 8px;">{row["증감률"]}</td><td style="padding: 8px; {color}">{row["가맹점수"]}</td></tr>'
            html += '</table>'
            
            st.markdown(html, unsafe_allow_html=True)
    
    # 하단 컨테이너
    bottom_container = st.container()
    with bottom_container:
        col1, col2 = st.columns(2)
        
        # 왼쪽: 생존 곡선 또는 위험 업종
        with col1:
            if selected_business != '전체':
                st.markdown("#### 생존 곡선 분석")
                
                if '개설일' in df.columns:
                    # 캐싱된 함수로 생존 분석 데이터 준비
                    unique_franchises_df = prepare_survival_data(df)
                    
                    if unique_franchises_df.empty:
                        st.info("생존분석에 필요한 데이터가 충분하지 않습니다.")
                    else:
                        fig = go.Figure()
                        
                        # X축 최대값을 저장할 변수
                        max_duration = 0
                        
                        occupation_df = unique_franchises_df[unique_franchises_df['업종'] == selected_business].copy()
                        
                        if not occupation_df.empty and len(occupation_df) > 1:
                            # 실제 데이터의 최대 운영기간
                            occupation_max = occupation_df['운영기간_개월'].max()
                            max_duration = max(max_duration, occupation_max)
                            
                            kmf_occupation = KaplanMeierFitter()
                            kmf_occupation.fit(
                                durations=occupation_df['운영기간_개월'],
                                event_observed=occupation_df['폐업여부']
                            )
                            
                            # survival_function을 실제 데이터 범위로 필터링
                            sf = kmf_occupation.survival_function_
                            sf_filtered = sf[sf.index <= occupation_max]
                            
                            fig.add_trace(go.Scatter(
                                x=sf_filtered.index,
                                y=sf_filtered['KM_estimate'],
                                mode='lines',
                                name=f'{selected_business} (전체지역)',
                                line=dict(color='#388e3c', width=2, dash='dash', shape='hv'),
                                hovertemplate='운영기간: %{x:.1f}개월<br>생존확률: %{y:.2%}<extra></extra>'
                            ))
                        
                        if selected_dong != '전체':
                            district_df = unique_franchises_df[unique_franchises_df['행정동'] == selected_dong].copy()
                            
                            if not district_df.empty and len(district_df) > 1:
                                district_max = district_df['운영기간_개월'].max()
                                max_duration = max(max_duration, district_max)
                                
                                kmf_district = KaplanMeierFitter()
                                kmf_district.fit(
                                    durations=district_df['운영기간_개월'],
                                    event_observed=district_df['폐업여부']
                                )
                                
                                sf = kmf_district.survival_function_
                                sf_filtered = sf[sf.index <= district_max]
                                
                                fig.add_trace(go.Scatter(
                                    x=sf_filtered.index,
                                    y=sf_filtered['KM_estimate'],
                                    mode='lines',
                                    name=f'{selected_dong} (전체업종)',
                                    line=dict(color='#f57c00', width=2, shape='hv'),
                                    hovertemplate='운영기간: %{x:.1f}개월<br>생존확률: %{y:.2%}<extra></extra>'
                                ))
                            
                            district_occupation_df = unique_franchises_df[
                                (unique_franchises_df['행정동'] == selected_dong) & 
                                (unique_franchises_df['업종'] == selected_business)
                            ].copy()
                            
                            if not district_occupation_df.empty and len(district_occupation_df) > 1:
                                district_occupation_max = district_occupation_df['운영기간_개월'].max()
                                max_duration = max(max_duration, district_occupation_max)
                                
                                kmf_district_occupation = KaplanMeierFitter()
                                kmf_district_occupation.fit(
                                    durations=district_occupation_df['운영기간_개월'],
                                    event_observed=district_occupation_df['폐업여부']
                                )
                                
                                sf = kmf_district_occupation.survival_function_
                                sf_filtered = sf[sf.index <= district_occupation_max]
                                
                                fig.add_trace(go.Scatter(
                                    x=sf_filtered.index,
                                    y=sf_filtered['KM_estimate'],
                                    mode='lines',
                                    name=f'{selected_dong} × {selected_business}',
                                    line=dict(color='#d32f2f', width=2, shape='hv'),
                                    hovertemplate='운영기간: %{x:.1f}개월<br>생존확률: %{y:.2%}<extra></extra>'
                                ))
                                
                                total_count = len(district_occupation_df)
                                closed_count = int((district_occupation_df['폐업여부'] == 1).sum())
                                survival_rate = 1 - (closed_count / total_count) if total_count > 0 else 0
                                
                                st.caption(f"📊 {selected_dong} × {selected_business}: 총 {total_count}개 가맹점, 폐업 {closed_count}개, 생존율 {survival_rate:.1%}")
                        
                        if fig.data:
                            fig.update_layout(
                                xaxis_title="운영 개월 수",
                                yaxis_title="생존 확률",
                                showlegend=True,
                                height=300,
                                margin=dict(t=40, b=40),
                                paper_bgcolor='white',
                                plot_bgcolor='white',
                                font=dict(color='#1a1a1a'),
                                yaxis=dict(range=[0, 1.05], tickformat='.0%', gridcolor='rgba(0,0,0,0.1)'),
                                xaxis=dict(
                                    range=[0, max_duration * 1.05],
                                    gridcolor='rgba(0,0,0,0.1)'
                                ),
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="right",
                                    x=0.99,
                                    bgcolor='rgba(255,255,255,0.8)',
                                    font=dict(size=10),
                                    bordercolor='#e0e0e0',
                                    borderwidth=1
                                ),
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("생존 곡선을 그릴 수 있는 데이터가 충분하지 않습니다.")
                else:
                    st.info("생존 분석에 필요한 데이터(개설일)가 없습니다.")
                    
            else:
                st.markdown("#### 위기경보 위험 업종")
                
                if selected_dong != '전체':
                    danger_df = current_df[current_df['위기경보등급'].isin(['위험', '주의'])]
                    
                    if not danger_df.empty:
                        business_danger = []
                        for business in current_df['업종'].unique():
                            business_df = current_df[current_df['업종'] == business]
                            danger_count = len(business_df[business_df['위기경보등급'].isin(['위험', '주의'])])
                            total_count = len(business_df)
                            if total_count > 0:
                                danger_ratio = (danger_count / total_count) * 100
                                business_danger.append({
                                    '업종': business,
                                    '위험비율': danger_ratio,
                                    '위험가맹점수': danger_count
                                })
                        
                        if business_danger:
                            danger_result = pd.DataFrame(business_danger).sort_values('위험비율', ascending=False)
                            
                            n = min(5, len(danger_result))
                            top_danger = danger_result.head(n)
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=top_danger['업종'],
                                    y=top_danger['위험비율'],
                                    text=[f"{x:.1f}%" for x in top_danger['위험비율']],
                                    textposition='outside',
                                    marker_color=['#d32f2f' if i == 0 else '#f57c00' for i in range(len(top_danger))]
                                )
                            ])
                            
                            fig.update_layout(
                                xaxis_title="",
                                yaxis_title="위험 비율 (%)",
                                showlegend=False,
                                height=300,
                                margin=dict(t=40, b=40),
                                paper_bgcolor='white',
                                plot_bgcolor='white',
                                font=dict(color='#1a1a1a'),
                                xaxis=dict(tickangle=-45),
                                yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("위험 업종이 없습니다.")
                    else:
                        st.info("위험 등급 데이터가 없습니다.")
                else:
                    st.info("행정동을 선택해주세요.")
        
        # 오른쪽: 매출등급 비교 또는 위험 업종
        with col2:
            if selected_business != '전체':
                st.markdown("#### 매출 등급별 핵심 지표 비교")
                
                business_df = current_df[current_df['업종'] == selected_business]
                
                if not business_df.empty:
                    low_sales = business_df[business_df['매출금액 구간'].isin([5, 6])]
                    other_sales = business_df[~business_df['매출금액 구간'].isin([5, 6])]
                    
                    if not low_sales.empty and not other_sales.empty:
                        metrics = []
                        
                        if '객단가 구간' in business_df.columns:
                            metrics.append({
                                '지표': '객단가 구간',
                                '하위등급(5,6)': low_sales['객단가 구간'].mean(),
                                '그 외 등급': other_sales['객단가 구간'].mean()
                            })
                        
                        if '취소율 구간' in business_df.columns:
                            metrics.append({
                                '지표': '취소율 구간',
                                '하위등급(5,6)': low_sales['취소율 구간'].mean(),
                                '그 외 등급': other_sales['취소율 구간'].mean()
                            })
                        
                        if '재방문 고객 비중' in business_df.columns:
                            metrics.append({
                                '지표': '재방문 고객 비중',
                                '하위등급(5,6)': low_sales['재방문 고객 비중'].mean(),
                                '그 외 등급': other_sales['재방문 고객 비중'].mean()
                            })
                        
                        if '유니크 고객 수 구간' in business_df.columns:
                            metrics.append({
                                '지표': '유니크 고객 수',
                                '하위등급(5,6)': low_sales['유니크 고객 수 구간'].mean(),
                                '그 외 등급': other_sales['유니크 고객 수 구간'].mean()
                            })
                        
                        if '매출건수 구간' in business_df.columns:
                            metrics.append({
                                '지표': '매출건수 구간',
                                '하위등급(5,6)': low_sales['매출건수 구간'].mean(),
                                '그 외 등급': other_sales['매출건수 구간'].mean()
                            })
                        
                        if metrics:
                            fig = go.Figure()
                            
                            for i, metric in enumerate(metrics):
                                fig.add_trace(go.Scatter(
                                    x=[metric['하위등급(5,6)'], metric['그 외 등급']],
                                    y=[i, i],
                                    mode='lines',
                                    line=dict(color='#999', width=2),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=[metric['하위등급(5,6)']],
                                    y=[i],
                                    mode='markers',
                                    marker=dict(size=12, color='#d32f2f'),
                                    name='하위등급(5,6)' if i == 0 else None,
                                    showlegend=True if i == 0 else False,
                                    hovertemplate=f"{metric['지표']}<br>하위등급: %{{x:.2f}}<extra></extra>"
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=[metric['그 외 등급']],
                                    y=[i],
                                    mode='markers',
                                    marker=dict(size=12, color='#388e3c'),
                                    name='그 외 등급' if i == 0 else None,
                                    showlegend=True if i == 0 else False,
                                    hovertemplate=f"{metric['지표']}<br>그 외: %{{x:.2f}}<extra></extra>"
                                ))
                                
                                gap = metric['그 외 등급'] - metric['하위등급(5,6)']
                                mid_point = (metric['하위등급(5,6)'] + metric['그 외 등급']) / 2
                                fig.add_annotation(
                                    x=mid_point,
                                    y=i,
                                    text=f"Gap: {gap:+.1f}",
                                    showarrow=False,
                                    yshift=15,
                                    font=dict(size=10, color='#1a1a1a')
                                )
                            
                            fig.update_layout(
                                yaxis=dict(
                                    ticktext=[m['지표'] for m in metrics],
                                    tickvals=list(range(len(metrics))),
                                    showgrid=False
                                ),
                                xaxis=dict(
                                    title="지표 값",
                                    showgrid=True,
                                    gridcolor='rgba(0,0,0,0.1)'
                                ),
                                height=300,
                                showlegend=True,
                                legend=dict(x=0.7, y=1, bgcolor='rgba(255,255,255,0.8)', bordercolor='#e0e0e0', borderwidth=1),
                                margin=dict(t=40, b=40, l=100, r=40),
                                paper_bgcolor='white',
                                plot_bgcolor='white',
                                font=dict(color='#1a1a1a')
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("비교할 지표 데이터가 없습니다.")
                    else:
                        st.info("매출 등급별 데이터가 충분하지 않습니다.")
                else:
                    st.info("선택한 업종의 데이터가 없습니다.")
                    
            else:
                st.markdown("#### 매출등급 위험 업종")
                
                if selected_dong != '전체':
                    low_sales_df = current_df[current_df['매출금액 구간'].isin([5, 6])]
                    
                    if not low_sales_df.empty:
                        business_sales = []
                        for business in current_df['업종'].unique():
                            business_df = current_df[current_df['업종'] == business]
                            low_count = len(business_df[business_df['매출금액 구간'].isin([5, 6])])
                            total_count = len(business_df)
                            if total_count > 0:
                                low_ratio = (low_count / total_count) * 100
                                business_sales.append({
                                    '업종': business,
                                    '위험비율': low_ratio,
                                    '위험가맹점수': low_count
                                })
                        
                        if business_sales:
                            sales_result = pd.DataFrame(business_sales).sort_values('위험비율', ascending=False)
                            
                            n = min(5, len(sales_result))
                            top_sales = sales_result.head(n)
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=top_sales['업종'],
                                    y=top_sales['위험비율'],
                                    text=[f"{x:.1f}%" for x in top_sales['위험비율']],
                                    textposition='outside',
                                    marker_color=['#d32f2f' if i == 0 else '#f57c00' for i in range(len(top_sales))]
                                )
                            ])
                            
                            fig.update_layout(
                                xaxis_title="",
                                yaxis_title="위험 비율 (%)",
                                showlegend=False,
                                height=300,
                                margin=dict(t=40, b=40),
                                paper_bgcolor='white',
                                plot_bgcolor='white',
                                font=dict(color='#1a1a1a'),
                                xaxis=dict(tickangle=-45),
                                yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("매출 위험 업종이 없습니다.")
                    else:
                        st.info("매출 위험 등급 데이터가 없습니다.")
                else:
                    st.info("행정동을 선택해주세요.")

if __name__ == "__main__":
    main()
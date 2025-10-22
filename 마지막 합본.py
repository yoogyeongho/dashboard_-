# -*- coding: utf-8 -*-
# 성동구 가맹점 현황 대시보드 (final_df 단일 데이터셋 버전)
# 실행: streamlit run app_final_dashboard.py

import re
import json
import math
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, List, Dict
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# --------------------------------------------
# 페이지/스타일
# --------------------------------------------
st.set_page_config(
    page_title="성동구 가맹점 현황 대시보드",
    layout="wide",
    page_icon="🏪",
    initial_sidebar_state="collapsed",  # ✅ 기본으로 사이드바 접기
)

PRIMARY = "#2563eb"  # blue-600
SAFE = "#16a34a"     # green-600
NORMAL = "#0ea5e9"   # sky-500
CAUTION = "#f59e0b"  # amber-500
DANGER = "#ef4444"   # red-500
GREY = "#6b7280"     # gray-500

st.markdown(
    """
    <style>
    .metric-good .stMetric {background: #ecfdf5; border-radius: 16px; padding: 8px 12px;}
    .metric-warn .stMetric {background: #fffbeb; border-radius: 16px; padding: 8px 12px;}
    .metric-bad  .stMetric {background: #fef2f2; border-radius: 16px; padding: 8px 12px;}
    .metric-neutral .stMetric {background: #f3f4f6; border-radius: 16px; padding: 8px 12px;}
    div[data-baseweb="select"] > div {border-radius: 10px;}
    .sep {margin: 8px 0 20px 0; border-top: 1px solid #e5e7eb;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------
# 유틸(지도/도형/포맷/노선집계)
# --------------------------------------------
def meters_per_pixel(lat: float, zoom: float) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)

def _deg_offset(lat: float, dx_m: float, dy_m: float) -> Tuple[float, float]:
    dlat = dy_m / 110_574.0
    dlon = dx_m / (111_320.0 * math.cos(math.radians(lat)))
    return dlon, dlat

def _square_polygon(lon: float, lat: float, half_m: float) -> List[List[float]]:
    dlon, dlat = _deg_offset(lat, half_m, half_m)
    return [
        [lon - dlon, lat - dlat],
        [lon + dlon, lat - dlat],
        [lon + dlon, lat + dlat],
        [lon - dlon, lat + dlat],
    ]

def _triangle_polygon(lon: float, lat: float, half_m: float) -> List[List[float]]:
    dlon_up, dlat_up = _deg_offset(lat, 0.0,  half_m)
    dlon_lw, dlat_lw = _deg_offset(lat, -half_m*0.866, -half_m*0.5)
    dlon_rw, dlat_rw = _deg_offset(lat,  half_m*0.866, -half_m*0.5)
    return [
        [lon + dlon_up, lat + dlat_up],
        [lon + dlon_rw, lat + dlat_rw],
        [lon + dlon_lw, lat + dlat_lw],
    ]

def build_polygon_df(poi_df: pd.DataFrame, size_px: float, lat0: float, zoom: float,
                     shape: str = "triangle",
                     lon_col: str = "경도", lat_col: str = "위도") -> pd.DataFrame:
    if poi_df is None or poi_df.empty:
        return poi_df
    mpp = meters_per_pixel(lat0, zoom)
    half_m = float(size_px) * mpp
    polys = []
    for _, r in poi_df.iterrows():
        lon = float(r[lon_col]); lat = float(r[lat_col])
        polygon = _triangle_polygon(lon, lat, half_m) if shape == "triangle" else _square_polygon(lon, lat, half_m)
        row = dict(r); row["polygon"] = polygon
        polys.append(row)
    return pd.DataFrame(polys)

def _fmt_num(v):
    try:
        return f"{float(v):.1f}"
    except Exception:
        return str(v) if v is not None else ""

# --------------------------------------------
# 좌표/반경/중복 유틸
# --------------------------------------------
def _dedup_points_by_xy(df: pd.DataFrame, lon="경도", lat="위도", digits: int = 5) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    key = (df[lon].round(digits).astype(str) + "_" + df[lat].round(digits).astype(str))
    return df.loc[~key.duplicated()].reset_index(drop=True)

def filter_pois_within_radius(poi_df: Optional[pd.DataFrame],
                              stores_df: pd.DataFrame,
                              radius_m: float,
                              lon_col="경도", lat_col="위도") -> Optional[pd.DataFrame]:
    if poi_df is None or poi_df.empty or stores_df is None or stores_df.empty:
        return poi_df
    try:
        from sklearn.neighbors import BallTree
        store_coords = np.radians(stores_df[["lat", "lon"]].to_numpy(dtype=float))
        poi_coords   = np.radians(poi_df[[lat_col, lon_col]].to_numpy(dtype=float))
        tree = BallTree(poi_coords, metric="haversine")
        R = 6371000.0
        rad = float(radius_m) / R
        hits = set()
        ind_list = tree.query_radius(store_coords, r=rad)
        for inds in ind_list:
            for j in inds:
                hits.add(int(j))
        if not hits:
            return poi_df.iloc[0:0]
        sub = poi_df.iloc[sorted(list(hits))].copy()
        sub = _dedup_points_by_xy(sub, lon=lon_col, lat=lat_col, digits=5)
        return sub
    except Exception:
        # fallback: 경계박스 + 하버사인
        def haversine(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1; dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            return 2*6371000*np.arcsin(np.sqrt(a))
        keep_idx = set()
        for _, s in stores_df.iterrows():
            lat0, lon0 = float(s["lat"]), float(s["lon"])
            box = poi_df[
                (poi_df[lat_col].between(lat0-0.02, lat0+0.02)) &
                (poi_df[lon_col].between(lon0-0.02, lon0+0.02))
            ]
            if box.empty:
                continue
            d = haversine(box[lat_col].to_numpy(), box[lon_col].to_numpy(), lat0, lon0)
            hits = box.index[d <= radius_m].tolist()
            keep_idx.update(hits)
        if not keep_idx:
            return poi_df.iloc[0:0]
        sub = poi_df.loc[sorted(list(keep_idx))].copy()
        sub = _dedup_points_by_xy(sub, lon=lon_col, lat=lat_col, digits=5)
        return sub

# --------------------------------------------
# 공간 필터(GeoJSON 행정동)
# --------------------------------------------
def _filter_points_by_dong_geojson(
    df: Optional[pd.DataFrame],
    selected_dong: Optional[str],
    geojson: Optional[dict]
) -> Optional[pd.DataFrame]:
    if df is None or df.empty or not selected_dong or not geojson:
        return df
    def _dong_name(props: dict) -> Optional[str]:
        if not isinstance(props, dict):
            return None
        keys = ["행정동", "name", "adm_nm", "ADM_DR_NM", "ADM_NM", "동"]
        for k in keys:
            if k in props and props[k]:
                return str(props[k]).strip()
        return None
    try:
        from shapely.geometry import Point, shape
        from shapely.ops import unary_union
        polys = []
        for f in geojson.get("features", []):
            nm = _dong_name(f.get("properties", {}))
            if nm == str(selected_dong).strip():
                try:
                    polys.append(shape(f["geometry"]))
                except Exception:
                    pass
        if not polys:
            return df
        poly = unary_union(polys) if len(polys) > 1 else polys[0]
        poly = poly.buffer(1e-9)
        mask = [poly.intersects(Point(lon, lat)) for lon, lat in zip(df["경도"], df["위도"])]
        return df.loc[mask]
    except Exception:
        # geometry 파싱 실패 시 bbox 대용
        try:
            xs, ys = [], []
            def _collect(coords):
                if coords and isinstance(coords[0], (list, tuple)) and isinstance(coords[0][0], (float, int)):
                    for x, y in coords:
                        xs.append(x); ys.append(y)
                else:
                    for c in coords:
                        _collect(c)
            hit = False
            for f in geojson.get("features", []):
                nm = _dong_name(f.get("properties", {}))
                if nm == str(selected_dong).strip():
                    _collect(f["geometry"]["coordinates"]); hit = True
            if not hit or not xs:
                return df
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            return df[(df["경도"] >= minx) & (df["경도"] <= maxx) &
                      (df["위도"] >= miny) & (df["위도"] <= maxy)]
        except Exception:
            return df

# --------------------------------------------
# 데이터 로더 (final_df / bus / subway / geojson)
# --------------------------------------------
@st.cache_data(show_spinner=True)
def load_final_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 타입 정리
    for c in ["경도","위도","회복탄력성_점수","임대료 점수","소비액 점수","교통접근성 점수","경쟁과열"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "폐업여부" in df.columns:
        df["폐업여부"] = pd.to_numeric(df["폐업여부"], errors="coerce").fillna(0).astype(int)
    if "기준년월" in df.columns:
        df["기준년월"] = df["기준년월"].astype(str)
    req = {"가맹점구분번호","경도","위도","행정동","업종"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    return df

@st.cache_data(show_spinner=True)
def load_bus(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        rename = {}
        if "X좌표" in df.columns: rename["X좌표"] = "경도"
        if "Y좌표" in df.columns: rename["Y좌표"] = "위도"
        df = df.rename(columns=rename)
        df["경도"] = pd.to_numeric(df["경도"], errors="coerce")
        df["위도"] = pd.to_numeric(df["위도"], errors="coerce")
        df = df.dropna(subset=["경도", "위도"])
        return df[["위도", "경도"] + [c for c in df.columns if c not in ["위도", "경도"]]]
    except Exception as e:
        st.warning(f"버스 데이터 로드 실패: {e}")
        return None

@st.cache_data(show_spinner=True)
def load_subway(path: str) -> Optional[pd.DataFrame]:
    try:
        if path.lower().endswith(".xlsx"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        rename = {}
        for c in df.columns:
            cl = str(c).lower().replace(" ", "")
            if cl in ["x좌표", "x"]:
                rename[c] = "위도"
            if cl in ["y좌표", "y"]:
                rename[c] = "경도"
        df = df.rename(columns=rename)
        df["경도"] = pd.to_numeric(df["경도"], errors="coerce")
        df["위도"] = pd.to_numeric(df["위도"], errors="coerce")
        df = df.dropna(subset=["경도", "위도"])
        return df[["위도", "경도"] + [c for c in df.columns if c not in ["위도", "경도"]]]
    except Exception as e:
        st.warning(f"지하철 데이터 로드 실패: {e}")
        return None

@st.cache_data(show_spinner=True)
def load_geojson(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# XGBoost 모델 학습 함수 
@st.cache_resource
def train_xgboost_model(df_full):
    """final_df 데이터로 XGBoost 모델 학습"""
    df_model = df_full.copy()

    # 기준년월 처리 및 푸리에 변환
    if '기준년월' in df_model.columns:
        df_model['기준년월'] = pd.to_datetime(df_model['기준년월'].astype(str), format='%Y%m', errors='coerce')
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

    # 2023년 데이터로만 학습 (없으면 전체로 fallback)
    if '기준년' in df_model.columns and (df_model['기준년'] == 2023).any():
        train_mask = df_model['기준년'] == 2023
    else:
        train_mask = np.ones(len(df_model), dtype=bool)

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]

    # scale_pos_weight 계산
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    # 모델 학습
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

    # Feature 준비(누락 피처는 0으로 채움)
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

    probabilities = _model.predict_proba(X_pred)[:, 1]
    return probabilities


@st.cache_data
def calculate_warning_grade(df, _model=None, features=None, _le_dict=None, _model_available=True):
    """폐업 확률 기반 위기경보 등급 계산 (캐싱)"""
    df = df.copy()

    if _model_available and _model is not None:
        # 폐업 확률 예측
        closure_probs = predict_closure_probability(df, _model, features, _le_dict)
        df['폐업확률'] = closure_probs

        # 하이브리드(절대+상대) 임계치
        absolute_thresholds = {'위험': 0.6, '주의': 0.4, '관심': 0.2, '안정': 0.0}
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

        # ✅ 대시보드 기존 컬럼명 호환: '위험등급', '위험도'도 같이 만들어 줌
        df['위험등급'] = df['위기경보등급']
        df['위험도']   = (df['폐업확률'] * 100.0).astype(float)

    else:
        # Fallback: 회복탄력성 분위 기반
        percentiles = df['회복탄력성_점수'].quantile([0.25, 0.5, 0.75]).tolist()
        df['위기경보등급'] = pd.cut(
            df['회복탄력성_점수'],
            bins=[-np.inf] + percentiles + [np.inf],
            labels=['위험', '주의', '관심', '안정']
        )
        df['위험등급'] = df['위기경보등급']
        df['위험도']   = 100.0 - pd.to_numeric(df.get('회복탄력성_점수', 50), errors='coerce').fillna(50)

    return df
    

# --------------------------------------------
# 스냅샷(매장별 '최근' 1행 선택) + 위험도/등급
# --------------------------------------------
def make_latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "기준년월" in work.columns:
        # YYYY-MM or YYYYMM 모두 처리
        try:
            work["_ym"] = work["기준년월"].astype(str).str.replace("-","", regex=False).astype(int)
        except Exception:
            work["_ym"] = pd.to_numeric(work["기준년월"], errors="coerce")
        work = work.sort_values(["가맹점구분번호","_ym"])
        snap = work.groupby("가맹점구분번호", as_index=False).tail(1).copy()
    else:
        # 기준년월 없으면 그대로 유니크 최근 한 줄 가정
        snap = work.drop_duplicates(subset=["가맹점구분번호"], keep="last").copy()

    # # 회복탄력성 점수 -> 위험도/등급
    snap["회복탄력성 점수"] = pd.to_numeric(snap.get("회복탄력성_점수", np.nan), errors="coerce")
    # s = snap["회복탄력성 점수"].dropna()
    # if len(s) >= 10:
    #     q1, q3 = np.percentile(s, [25, 75])
    #     iqr = q3 - q1
    #     low_out = q1 - 1.5 * iqr
    #     high_out = q3 + 1.5 * iqr
    #     def risk_grade(v: float) -> str:
    #         if np.isnan(v): return "낮음"
    #         if v <= low_out: return "위험"
    #         elif v <= q1: return "주의"
    #         elif v <= q3: return "보통"
    #         elif v <= high_out: return "안전"
    #         else: return "매우안전"
    # else:
    #     def risk_grade(v: float) -> str:
    #         if np.isnan(v): return "낮음"
    #         if v < 25: return "위험"
    #         elif v < 50: return "주의"
    #         elif v < 75: return "보통"
    #         else: return "안전"

    # snap["위험등급"] = snap["회복탄력성 점수"].apply(risk_grade)
    # snap["위험도"] = 100.0 - snap["회복탄력성 점수"].fillna(50.0)

    # 좌표 숫자화/결측 제거
    snap["lon"] = pd.to_numeric(snap["경도"], errors="coerce")
    snap["lat"] = pd.to_numeric(snap["위도"], errors="coerce")
    snap = snap.dropna(subset=["lon","lat"])
    return snap

# --------------------------------------------
# KPI 계산
# --------------------------------------------
def kpi_summary(df_snapshot: pd.DataFrame) -> Dict[str, float]:
    total = len(df_snapshot)
    pr_caution = float((df_snapshot["위험등급"].isin(["주의","위험"]).mean()) * 100.0) if "위험등급" in df_snapshot else np.nan
    closed_rate = float((df_snapshot["폐업여부"] == 1).mean() * 100.0) if "폐업여부" in df_snapshot else np.nan
    mean_res = float(df_snapshot["회복탄력성 점수"].mean()) if "회복탄력성 점수" in df_snapshot else np.nan
    return {
        "전체 가맹점 수": total,
        "주의/위험 비중(%)": pr_caution,
        "폐업 매장 비중(%)": closed_rate,
        "평균 회복탄력성": mean_res,
    }

# --------------------------------------------
# 사이드바
# --------------------------------------------
# with st.sidebar:
#     st.header("📁 데이터 경로")
#     final_path  = st.text_input("final_df: final_df.zip", value="final_df.zip")
#     bus_path    = st.text_input("버스: merged_bus_data 최종ver.csv", value="merged_bus_data 최종ver.csv")
#     subway_path = st.text_input("지하철: 성동구 내 역사 및 인근 역 정제된 파일.xlsx", value="성동구 내 역사 및 인근 역 정제된 파일.xlsx")
#     geojson_path= st.text_input("행정동 GeoJSON (선택)", value="seongdong_행정동.geojson")
#     st.caption("경로가 다르면 수정하세요. 불러오기 실패 시 해당 레이어는 자동으로 숨김 처리됩니다.")

#     show_bus = st.checkbox("버스 정류장 표시", value=True, key="opt_bus")
#     show_subway = st.checkbox("지하철 역 표시", value=True, key="opt_subway")

#     st.markdown("---")
#     st.subheader("🗺️ 마커 크기 설정")
#     size_mode = st.selectbox("마커 크기 모드", ["고정", "위험도 비례"], index=0, key="size_mode")
#     marker_px = st.slider("기본 마커 크기(픽셀)", 2, 20, 6, key="marker_px")
#     marker_min = st.slider("최소 크기(비례)", 2, 16, 3, key="marker_min")
#     marker_max = st.slider("최대 크기(비례)", 6, 28, 8, key="marker_max")
# --------------------------------------------
# Config (사이드바 제거 : 고정값)
# --------------------------------------------
final_path   = "final_df.zip"
bus_path     = "merged_bus_data 최종ver.csv"
subway_path  = "성동구 내 역사 및 인근 역 정제된 파일.xlsx"
geojson_path = "seongdong_행정동.geojson"
# 레이어 표시 옵션
show_bus    = True
show_subway = True
# 마커 크기 옵션
size_mode  = "고정"   # "고정" 또는 "위험도 비례"
marker_px  = 6
marker_min = 3
marker_max = 8

# --------------------------------------------
# 데이터 적재
# --------------------------------------------
final_df = load_final_df(final_path)
bus_df = load_bus(bus_path)
subway_df = load_subway(subway_path)
geojson = load_geojson(geojson_path)

# 👉 모델 학습 (캐싱됨)
model, model_features, model_le_dict = train_xgboost_model(final_df)

# --------------------------------------------
# 헤더/타이틀
# --------------------------------------------
st.title("🏪 성동구 가맹점 현황 대시보드")
st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

# --------------------------------------------
# 스냅샷 생성(최신월 기준 매장 1행)
# --------------------------------------------
snapshot = make_latest_snapshot(final_df)
snapshot = calculate_warning_grade(snapshot, _model=model, features=model_features, _le_dict=model_le_dict, _model_available=True)


# --------------------------------------------
# 필터 영역 (행정동/업종)
# --------------------------------------------
col_f1, col_f2 = st.columns([1,1])

with col_f1:
    dong_opts = ["(전체)"] + sorted(snapshot["행정동"].dropna().astype(str).unique().tolist())
    selected_dong = st.selectbox("지역(행정동)", options=dong_opts)
    if selected_dong == "(전체)":
        selected_dong = None

with col_f2:
    if selected_dong:
        cats = sorted(snapshot[snapshot["행정동"]==selected_dong]["업종"].dropna().unique().tolist())
    else:
        cats = sorted(snapshot["업종"].dropna().unique().tolist())
    selected_category = st.selectbox("업종", options=["(전체)"] + cats)
    if selected_category == "(전체)":
        selected_category = None

# 필터 적용
filtered = snapshot.copy()
if selected_dong:
    filtered = filtered[filtered["행정동"] == selected_dong]
if selected_category:
    filtered = filtered[filtered["업종"] == selected_category]

# --------------------------------------------
# KPI 섹션 (전체 KPI)
# --------------------------------------------
st.subheader("📊 KPI")
# CSS 스타일
st.markdown("""
<style>
.kpi-card {
    border: 2px solid #E5E7EB;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    background-color: #F9FAFB;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.kpi-title {
    font-weight: 600;
    font-size: 0.9rem;
    color: #374151;
    margin-bottom: 6px;
}
.kpi-value {
    font-weight: 700;
    font-size: 1.4rem;
    color: #111827;
}
</style>
""", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
all_kpi = kpi_summary(snapshot)
col1.markdown(f"""
<div class="kpi-card">
  <div class="kpi-title">전체 가맹점 수</div>
  <div class="kpi-value">{all_kpi['전체 가맹점 수']:,}</div>
</div>
""", unsafe_allow_html=True)
col2.markdown(f"""
<div class="kpi-card">
  <div class="kpi-title">주의/위험 비중</div>
  <div class="kpi-value">{all_kpi['주의/위험 비중(%)']:.1f}%</div>
</div>
""", unsafe_allow_html=True)
col3.markdown(f"""
<div class="kpi-card">
  <div class="kpi-title">폐업 매장 비중</div>
  <div class="kpi-value">{all_kpi['폐업 매장 비중(%)']:.1f}%</div>
</div>
""", unsafe_allow_html=True)
col4.markdown(f"""
<div class="kpi-card">
  <div class="kpi-title">평균 회복탄력성</div>
  <div class="kpi-value">{all_kpi['평균 회복탄력성']:.1f}</div>
</div>
""", unsafe_allow_html=True)
# --------------------------------------------
# 타겟 KPI (필터 반영)
# --------------------------------------------
st.subheader("🎯 타겟 KPI")
colf1, colf2, colf3, colf4 = st.columns(4)
if filtered.empty or (selected_dong is None and selected_category is None):
    for c, title in zip([colf1, colf2, colf3, colf4],
                        ["선택 가맹점 수", "주의/위험 비중", "폐업 매장 비중", "평균 회복탄력성"]):
        c.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">-</div>
        </div>
        """, unsafe_allow_html=True)
else:
    k = kpi_summary(filtered)
    vals = [
        f"{k['전체 가맹점 수']:,}",
        f"{k['주의/위험 비중(%)']:.1f}%",
        f"{k['폐업 매장 비중(%)']:.1f}%",
        f"{k['평균 회복탄력성']:.1f}"
    ]
    for c, title, val in zip(
        [colf1, colf2, colf3, colf4],
        ["선택 가맹점 수", "주의/위험 비중", "폐업 매장 비중", "평균 회복탄력성"],
        vals
    ):
        c.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{val}</div>
        </div>
        """, unsafe_allow_html=True)
st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
# --------------------------------------------
# 지도 섹션 (pydeck) - 기존 로직 유지
# --------------------------------------------
st.subheader("🗺️ 가맹점 분포 지도")

map_df = filtered.copy()
if map_df.empty:
    st.info("선택된 조건에 해당하는 점포가 없습니다.")
else:
    # 색상
    color_map = {
    "위험": (214, 39, 40),
    "주의": (255, 127, 0),
    "관심": (255, 191, 0),
    "안정": (56, 168, 0),
}

    def _rgb_tuple(grade):
        t = color_map.get(str(grade), (120,120,120))
        return [int(t[0]), int(t[1]), int(t[2])]
    rgb = np.array([_rgb_tuple(g) for g in map_df["위험등급"].astype(object).tolist()], dtype=int)
    map_df[["r","g","b"]] = rgb
    map_df["a"] = 200

    if size_mode == "고정":
        map_df["pt_size"] = int(marker_px)
    else:
        _risk = pd.to_numeric(map_df.get("위험도", np.nan), errors="coerce").fillna(0.0).clip(0, 100)
        map_df["pt_size"] = (_risk / 100.0) * (marker_max - marker_min) + marker_min
        map_df["pt_size"] = map_df["pt_size"].astype(float)

# 가맹점 툴팁(텍스트)
if not map_df.empty:
    def _mk_store_tip(r):
        # 폐업여부 → 이름 옆에 표기
        closed_raw = r.get('폐업여부', None)
        try:
            status = "폐업" if int(closed_raw) == 1 else "영업"
        except Exception:
            status = "" if closed_raw is None else str(closed_raw)

        name = str(r.get('가맹점명', ''))
        # 필요시 영업일 땐 표시 안 하려면 아래 한 줄을: name_line = f"{name}" if status != "폐업" else f"{name} (폐업)"
        name_line = f"{name}" if not status else f"{name} ({status})"

        lines = [
            name_line,
            f"{r.get('업종','')} / {r.get('행정동','')}",
            f"회복탄력성: {_fmt_num(r.get('회복탄력성 점수'))} · "
            f"위험도: {_fmt_num(r.get('위험도'))} ({r.get('위험등급','')})",
            f"고객다양성: {_fmt_num(r.get('고객다양성_점수'))} · "
            f"고객인구 적합도: {_fmt_num(r.get('고객인구_적합도_점수'))}",
            f"임대료대비매출: {_fmt_num(r.get('임대료대비매출_점수'))} · "
            f"객단가 안정성: {_fmt_num(r.get('객단가_안정성_점수'))}",
            f"가맹점입지점수: {_fmt_num(r.get('가맹점입지점수'))}",
        ]
        return "\n".join(lines)

    map_df["tooltip"] = map_df.apply(_mk_store_tip, axis=1)



    # 초기 뷰
    init_lat = float(map_df["lat"].mean())
    init_lng = float(map_df["lon"].mean())
    view_state = pdk.ViewState(latitude=init_lat, longitude=init_lng, zoom=12.5, pitch=0)

    # 레이어: 가맹점
    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[lon, lat]',
            get_fill_color='[r, g, b, a]',
            get_radius='pt_size',
            radiusUnits='pixels',
            radiusMinPixels=1,
            radiusMaxPixels=int(marker_max) + 4,
            stroked=True,
            get_line_color=[0, 0, 0, 120],
            lineWidthMinPixels=1,
            pickable=True,
            auto_highlight=True,
        )
    ]

    # ------------------------------------------------------------
    # 버스/지하철 표시 대상 만들기
    #   - 필터 없음(행정동/업종 모두 전체): 전체 데이터 그대로 표시
    #   - 필터 있음(행정동 또는 업종 선택): 기존 규칙 유지
    #       · 행정동 선택 시 GeoJSON 교집합
    #       · 가맹점 반경 필터(버스 400m, 지하철 1000m)
    # ------------------------------------------------------------
    is_filtered = bool(selected_dong or selected_category)

    # 1) 기본 후보 집합
    if selected_dong and geojson is not None:
        # 행정동을 선택한 경우에만 GeoJSON 교집합 적용
        bus_df_for_map    = _filter_points_by_dong_geojson(bus_df, selected_dong, geojson)
        subway_df_for_map = _filter_points_by_dong_geojson(subway_df, selected_dong, geojson)
    else:
        # 필터가 없거나(전체), GeoJSON이 없으면 원본 전체
        bus_df_for_map    = bus_df.copy() if bus_df is not None else None
        subway_df_for_map = subway_df.copy() if subway_df is not None else None

    # 2) 반경 필터는 "필터가 활성화된 경우"에만 적용
    if is_filtered:
        if show_bus and bus_df_for_map is not None and not bus_df_for_map.empty:
            bus_df_for_map = filter_pois_within_radius(
                bus_df_for_map,
                map_df[["lat","lon"]],
                radius_m=400.0
            )
        if show_subway and subway_df_for_map is not None and not subway_df_for_map.empty:
            subway_df_for_map = filter_pois_within_radius(
                subway_df_for_map,
                map_df[["lat","lon"]],
                radius_m=1000.0
            )
    # 필터가 비활성화(전체 보기)면 반경 필터를 건너뛴다 → 전체 그대로 표시


    # 버스 툴팁/도형
    if show_bus and bus_df_for_map is not None and not bus_df_for_map.empty:
        for col in ["간선","지선","순환","마을"]:
            if col not in bus_df_for_map.columns:
                bus_df_for_map[col] = ""
        def _find_list_col(df, key):
            patterns = ["노선", "목록", "리스트", "list", "routes"]
            cands = [c for c in df.columns if (key in str(c)) and any(p in str(c) for p in patterns)]
            return sorted(cands, key=len, reverse=True)[0] if cands else None
        list_cols = {k:_find_list_col(bus_df_for_map, k) for k in ["간선","지선","순환","마을"]}
        def _route_count(cell):
            if cell is None or (isinstance(cell, float) and np.isnan(cell)): return 0
            s = str(cell).strip()
            if not s or s in ["-","nan","None"]: return 0
            parts = re.split(r"[,\s;/|·\[\]\(\)\'\"]+|\n|\r", s)
            parts = [p for p in parts if p]
            return len(parts)
        def _count_for_key(df, key):
            list_col = list_cols.get(key)
            base_col = key if key in df.columns else None
            if list_col:
                return df[list_col].apply(_route_count)
            if base_col:
                ser = pd.to_numeric(df[base_col], errors="coerce")
                if ser.notna().any():
                    if ser.max() <= 1: return ser.fillna(0).astype(int)
                    return ser.fillna(0).astype(int)
                return df[base_col].apply(_route_count)
            return pd.Series(0, index=df.index, dtype=int)
        bus_df_for_map["_n_간선"] = _count_for_key(bus_df_for_map, "간선")
        bus_df_for_map["_n_지선"] = _count_for_key(bus_df_for_map, "지선")
        bus_df_for_map["_n_순환"] = _count_for_key(bus_df_for_map, "순환")
        bus_df_for_map["_n_마을"] = _count_for_key(bus_df_for_map, "마을")
        total_col = None
        for c in bus_df_for_map.columns:
            if any(k in str(c) for k in ["총노선", "노선수"]):
                total_col = c; break
        if total_col:
            bus_df_for_map["_n_total"] = pd.to_numeric(bus_df_for_map[total_col], errors="coerce").fillna(0).astype(int)
        else:
            bus_df_for_map["_n_total"] = bus_df_for_map[["_n_간선","_n_지선","_n_순환","_n_마을"]].sum(axis=1)

        if "정류소명_x" in bus_df_for_map.columns:
            name_col = "정류소명_x"
        elif "정류소명" in bus_df_for_map.columns:
            name_col = "정류소명"
        else:
            name_col = "_tmp_name"; bus_df_for_map[name_col] = "버스정류장"

        def _mk_bus_tip(r):
            return (f"🚌 {r.get(name_col,'')}\n"
                    f"간선 {int(r.get('_n_간선',0))} · "
                    f"지선 {int(r.get('_n_지선',0))} · "
                    f"순환 {int(r.get('_n_순환',0))} · "
                    f"마을 {int(r.get('_n_마을',0))} "
                    f"(총 {int(r.get('_n_total',0))})")
        bus_df_for_map["tooltip"] = bus_df_for_map.apply(_mk_bus_tip, axis=1)

        # 크기: 가맹점 대비 작게
        poi_px = int(np.clip((map_df["pt_size"].median() if ("pt_size" in map_df.columns and not map_df["pt_size"].empty) else marker_px), 6, 32))
        poi_px_small = max(0.3, poi_px * 0.0001)  # 기존 로직 유지
        bus_poly_df = build_polygon_df(bus_df_for_map, size_px=poi_px_small, lat0=init_lat, zoom=view_state.zoom, shape="triangle")
        bus_layer = pdk.Layer(
            "PolygonLayer",
            data=bus_poly_df,
            get_polygon="polygon",
            filled=True,
            get_fill_color=[0, 0, 0, 220],
            stroked=True,
            get_line_color=[0, 0, 0, 255],
            lineWidthMinPixels=1,
            pickable=True,
            auto_highlight=True,
        )
        layers.append(bus_layer)

    # 지하철 툴팁/도형
    if show_subway and subway_df_for_map is not None and not subway_df_for_map.empty:
        for need in ["역명","호선","역번호"]:
            if need not in subway_df_for_map.columns:
                subway_df_for_map[need] = ""
        def _normalize_line(v):
            s = str(v).strip()
            if not s: return ""
            if re.fullmatch(r"\d+", s):
                s = f"{int(s)}호선"
            elif ("호선" not in s) and not re.search(r"[가-힣]호선$", s):
                s = f"{s}호선"
            return s
        def _mk_sub_tip(r):
            line = _normalize_line(r.get("호선",""))
            sta  = str(r.get("역명","")).strip()
            no   = str(r.get("역번호","")).strip()
            bits = []
            if line: bits.append(f"호선 : {line}")
            if no:   bits.append(f"역번호 : {no}")
            info = " , ".join(bits)
            return f"🚇 {sta}\n{info}"
        subway_df_for_map["tooltip"] = subway_df_for_map.apply(_mk_sub_tip, axis=1)

        poi_px = int(np.clip((map_df["pt_size"].median() if ("pt_size" in map_df.columns and not map_df["pt_size"].empty) else marker_px), 6, 32))
        poi_px_small = max(0.3, poi_px * 0.0001)
        subway_poly_df = build_polygon_df(subway_df_for_map, size_px=poi_px_small, lat0=init_lat, zoom=view_state.zoom, shape="square")
        subway_layer = pdk.Layer(
            "PolygonLayer",
            data=subway_poly_df,
            get_polygon="polygon",
            filled=True,
            get_fill_color=[0, 0, 0, 220],
            stroked=True,
            get_line_color=[0, 0, 0, 255],
            lineWidthMinPixels=1,
            pickable=True,
            auto_highlight=True,
        )
        layers.append(subway_layer)

# === 가로 한 줄 요약(가맹점/버스/지하철 표시 개수) ===
c1, c2, c3 = st.columns(3)

# 표시(Shown)는 실제 지도에 그려지는 개수로 계산
store_total = len(snapshot)
store_shown = len(map_df)

bus_total = len(bus_df) if bus_df is not None else 0
bus_shown = (len(bus_df_for_map) if (show_bus and (bus_df_for_map is not None)) else 0)

subway_total = len(subway_df) if subway_df is not None else 0
subway_shown = (len(subway_df_for_map) if (show_subway and (subway_df_for_map is not None)) else 0)

with c1:
    st.caption(f"🏪 가맹점: 전체 {store_total:,} → 표시 {store_shown:,}")
with c2:
    st.caption(f"🚌 버스 정류장: 전체 {bus_total:,} → 표시 {bus_shown:,}")
with c3:
    st.caption(f"🚇 지하철 역: 전체 {subway_total:,} → 표시 {subway_shown:,}")

# ✅ 여기서부터는 컬럼 블록 '밖'(들여쓰기 0)
TOOLTIP = {"text": "{tooltip}"}
st.pydeck_chart(
    pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=view_state,
        layers=layers,
        tooltip=TOOLTIP,
    ),
    use_container_width=True,  # 전체 폭 사용
)

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

# --------------------------------------------
# 위험 단계 매장 순위 (단일 선택 체크박스)
# --------------------------------------------
st.subheader("⚠️ 위험 단계 매장 순위")
rank_df = filtered.copy()
if selected_dong:
    rank_df = rank_df[rank_df["행정동"] == selected_dong]
if selected_category:
    rank_df = rank_df[rank_df["업종"] == selected_category]

rank_df = filtered.copy()
# ...
order = pd.CategoricalDtype(categories=["위험","주의","관심","안정"], ordered=True)
if "위험등급" in rank_df.columns:
    rank_df["위험등급"] = rank_df["위험등급"].astype(order)

rank_df = rank_df.sort_values(["위험등급","위험도"], ascending=[True, False]).reset_index(drop=True)


show_cols = [
    "가맹점구분번호","가맹점명","행정동","업종","위험도","위험등급","회복탄력성 점수",
    "교통접근성 점수","경쟁과열","폐업여부","임대료 점수","소비액 점수"
]
show_cols = [c for c in show_cols if c in rank_df.columns]
rank_df = rank_df.sort_values(["위험등급","위험도"], ascending=[True, False]).reset_index(drop=True)

# ✅ 현재 선택(가맹점구분번호)을 세션에 보관
if "sel_sid" not in st.session_state:
    st.session_state.sel_sid = None

# 테이블 표시용 DF 생성: '선택'은 현재 선택된 행만 True
rank_df_display = rank_df.copy()
rank_df_display.insert(
    0, "선택",
    rank_df_display["가맹점구분번호"].astype(str) == (st.session_state.sel_sid or "")
)

edited = st.data_editor(
    rank_df_display[["선택"] + show_cols],
    use_container_width=True,
    hide_index=True,
    key="rank_editor",
    column_config={
        "선택": st.column_config.CheckboxColumn(
            "선택",
            help="상세 분석할 매장을 한 개 선택하세요",
            default=False
        )
    }
)

# 단일 선택 강제 로직: 여러 개가 True여도 '마지막 의도'만 남기고 나머지는 해제
sel_mask = edited["선택"] == True
if sel_mask.any():
    selected_ids = edited.loc[sel_mask, "가맹점구분번호"].astype(str).tolist()
    prev = st.session_state.sel_sid
    # 이전 선택이 포함되어 있고 2개 이상이면, 이전 것 제외한 첫 번째를 새 선택으로
    if prev in selected_ids and len(selected_ids) > 1:
        new_sel = next((x for x in selected_ids if x != prev), prev)
    else:
        new_sel = selected_ids[0]
    if new_sel != prev:
        st.session_state.sel_sid = new_sel
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
else:
    # 모두 해제되면 선택 해제
    if st.session_state.sel_sid is not None:
        st.session_state.sel_sid = None
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

# 이후 섹션(매장 상세 분석)에서 사용할 선택값
sel_sid = st.session_state.sel_sid

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)


# --------------------------------------------
# 매장 상세 분석 (선택 매장) — 새 구성 (2섹션)
# --------------------------------------------
st.subheader("🔍 선택 매장 상세 분석")

if sel_sid is None:
    st.info("위의 **위험 단계 매장 순위** 표에서 매장을 하나 선택하면 상세 분석이 표시됩니다.")
else:
    # 선택 매장 단일 행 확보
    store_row = rank_df[rank_df["가맹점구분번호"].astype(str) == sel_sid]
    if store_row.empty:
        st.info("선택한 매장의 상세 정보를 찾을 수 없습니다.")
    else:
        st.write(f"**{store_row.iloc[0].get('가맹점명','')}** / {store_row.iloc[0].get('행정동','')} / {store_row.iloc[0].get('업종','')}")

        # --- 안전 가드: 선택 매장 시계열 확보 ---
        ts = final_df[final_df["가맹점구분번호"].astype(str) == sel_sid].copy()
        if ts.empty or ("기준년월" not in ts.columns):
            st.info("해당 매장의 시계열 데이터가 없거나 '기준년월' 컬럼이 없습니다.")
        else:
            # ===============================
            # 섹션 1) 2분할
            #   - 왼쪽: 6개 지표의 '분기별' 추이 (라인)
            #   - 오른쪽: 레이더(최신 분기)
            # ===============================

            # 기준년월 → 날짜 변환(YYYY-MM/ YYYYMM 모두 대응)
            def _parse_ym(val):
                s = str(val)
                if "-" in s:  # YYYY-MM
                    s2 = s + "-01" if len(s) == 7 else s
                else:         # YYYYMM
                    s2 = (s[:4] + "-" + s[4:6] + "-01") if len(s) >= 6 else s
                return pd.to_datetime(s2, errors="coerce")

            ts_q = ts.copy()
            ts_q["_date"] = ts_q["기준년월"].apply(_parse_ym)
            ts_q = ts_q.dropna(subset=["_date"]).sort_values("_date")

            # 사용할 6개 지표(존재하는 컬럼만)
            six_metrics = [
                "고객다양성_점수",
                "임대료대비매출_점수",
                "객단가_안정성_점수",
                "고객인구_적합도_점수",
                "가맹점입지점수",
                "소비액대비매출점수",
            ]
            six_metrics = [c for c in six_metrics if c in ts_q.columns]

            col_l, col_r = st.columns([2, 1])

            # --- (왼쪽) 분기별 라인 ---
            if six_metrics:
                # 분기 키
                qkey = ts_q["_date"].dt.to_period("Q")
                agg_dict = {c: "mean" for c in six_metrics}
                qdf = ts_q.groupby(qkey).agg(agg_dict).reset_index(names="분기")

                # 사람이 읽기 쉬운 라벨
                def _pretty_quarter(p):
                    s = str(p)  # e.g., '2024Q3'
                    try:
                        year = int(s[:4]); q = int(s[-1])
                        return f"{year}년 {q}분기"
                    except Exception:
                        return s

                qdf["분기라벨"] = qdf["분기"].apply(_pretty_quarter)

                if not qdf.empty:
                    q_long = qdf.melt(
                        id_vars=["분기", "분기라벨"],
                        value_vars=six_metrics,
                        var_name="지표",
                        value_name="값",
                    )
                    fig_quarter_lines = px.line(
                        q_long,
                        x="분기라벨",
                        y="값",
                        color="지표",
                        markers=True,
                        title="분기별 지표 추이(평균)",
                        hover_data={"분기라벨": False, "지표": True, "값": ":.1f"},
                        labels={"값": "점수(0~100)"}
                    )

                    # ✅ x축 라벨 없애기
                    fig_quarter_lines.update_xaxes(title_text="")

                    # 범례 위치 조정
                    fig_quarter_lines.update_layout(
                        height=360,
                        margin=dict(l=10, r=10, t=50, b=110),   # 🔽 범례 자리 확보
                        legend=dict(
                            orientation="h",
                            x=0.5, xanchor="center",            # 가로 중앙 정렬
                            y=-0.25, yanchor="top",             # 🔽 더 아래로 내림
                            bgcolor="rgba(0,0,0,0)"
                        ),
                        legend_title_text=""
                    )
                    # x축 라벨 숨김 (tick은 그대로)
                    fig_quarter_lines.update_xaxes(title_text="")
                    # 축 여백 자동 조정(선택)
                    fig_quarter_lines.update_xaxes(automargin=True)
                    fig_quarter_lines.update_yaxes(automargin=True)

                    col_l.plotly_chart(fig_quarter_lines, use_container_width=True)

                else:
                    col_l.info("분기별로 집계할 데이터가 충분하지 않습니다.")
            else:
                col_l.info("분기별 추이를 그릴 수 있는 지표가 없습니다.")

            # --- (오른쪽) 최신 분기 레이더 ---
            if six_metrics and 'qdf' in locals() and not qdf.empty:
                latest_q = qdf.iloc[-1]
                radar_vals = [
                    float(latest_q.get(c, np.nan)) if pd.notna(latest_q.get(c, np.nan)) else None
                    for c in six_metrics
                ]
                categories = [c.replace("_", "<br>") for c in six_metrics]

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=radar_vals,
                    theta=categories,
                    fill="toself",
                    name="최신 분기"
                ))
                fig_radar.update_layout(
                    title=f"레이더(최신 분기: {latest_q['분기라벨']})",
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                    height=360,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                col_r.plotly_chart(fig_radar, use_container_width=True)
            else:
                col_r.info("레이더를 그릴 수 있는 최신 분기 데이터가 없습니다.")

            st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

            # ===============================
            # 섹션 2) 전체 폭 — 월별 '매출' 추이
            # ===============================
            sales_candidates = ["매출금액", "매출구간", "총매출", "매출", "매출금액 구간", "매출건수"]
            sales_col = next((c for c in sales_candidates if c in ts_q.columns), None)

            if sales_col is not None:
                y_series = pd.to_numeric(ts_q[sales_col], errors="coerce")
                y_is_numeric = y_series.notna().any()

                # ✅ 월 라벨 포맷: '2023년 1월' 식으로
                ts_q["_label"] = ts_q["_date"].dt.strftime("%Y년 %#m월")  # 윈도우면 %-m 대신 %#m 사용

                fig_sales = px.line(
                    ts_q,
                    x="_label",
                    y=sales_col,
                    markers=True,
                    title=f"월별 {sales_col} 추이",
                    labels={"_label": "기준년월", sales_col: ("매출구간" if y_is_numeric else sales_col)},
                )

                # ✅ 전체 기간을 2023.1 ~ 2024.12로 고정
                start_date = pd.Timestamp("2023-01-01")
                end_date   = pd.Timestamp("2024-12-31")

                # Plotly layout 설정
                fig_sales.update_layout(
                    height=360,
                    margin=dict(l=10, r=10, t=50, b=10),
                    xaxis=dict(
                        title="기준년월",
                        tickmode="array",
                        tickvals=ts_q["_label"].unique().tolist(),
                        tickangle=45
                    ),
                    yaxis=dict(
                        title=("매출구간" if y_is_numeric else sales_col),
                        range=[1, 6],     # ✅ Y축 1~6 고정
                        dtick=1
                    )
                )

                st.plotly_chart(fig_sales, use_container_width=True)
            else:
                st.info("월 단위 매출 추이를 그릴 수 있는 컬럼을 찾지 못했습니다. (예: '매출금액', '매출금액 구간' 등)")


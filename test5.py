import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from datetime import datetime
import io

# ==============================================================================
# 1. 페이지 및 기본 설정
# ==============================================================================
st.set_page_config(page_title="여객노선부 연결 분석기", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 350px;
        max-width: 350px;
    }
    /* Bank View 버튼 텍스트 스타일링 (Bold & Size Up) */
    div.stButton > button {
        font-weight: bold !important;
        font-size: 15px !important;
        border: 1px solid #ddd;
    }
    div.stButton > button p {
        font-weight: bold !important;
        font-size: 15px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("연결 스케줄 분석 앱 VER.2.0")

# --- 모드 선택 (사이드바) ---
analysis_mode = st.sidebar.radio(
    "기능 모드 선택",
    ["스케줄 데이터 변환", "단일 스케줄 분석", "두 스케줄 비교 분석"]
)

# --- [NOTICE] 데이터 작성 가이드 ---
if analysis_mode != "스케줄 데이터 변환":
    with st.expander("[필독] 분석용 데이터(CSV) 작성 양식 가이드", expanded=False):
        st.markdown("""
        ##### 1. 필수 컬럼
        * **SEASON**: 시즌 (예: S26)
        * **FLT NO**: 편명 (예: '081')
        * **ORGN**: 출발지 공항
        * **DEST** (또는 DESTINATION): 도착지 공항
        * **STD / STA**: 시간 (HH:MM)
        * **OPS**: 항공사 코드
        * **ROUTE**: 노선 구분 (예: 미주노선, 동남아노선, CHN, JPN 등) -> **색상 구분 기준**
        * **구분**: `To ICN` (도착) / `From ICN` (출발)
        """)

# ==============================================================================
# 2. 공통 함수 정의
# ==============================================================================

@st.cache_data
def load_data(file):
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']
    for enc in encodings:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            df.columns = df.columns.str.strip()
            if 'DESTINATION' in df.columns:
                df.rename(columns={'DESTINATION': 'DEST'}, inplace=True)

            required = ['OPS', 'FLT NO', '구분', 'STD', 'STA', 'ORGN', 'DEST', 'ROUTE']
            if all(col in df.columns for col in required):
                for col in ['구분', 'FLT NO', 'ROUTE', 'OPS', 'ORGN', 'DEST']:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.strip()
                return df
        except:
            continue
    return None

def time_to_minutes(t_str):
    try:
        h, m = map(int, t_str.split(':'))
        return h * 60 + m
    except:
        return None

def get_time_slot(time_str):
    try:
        dt = datetime.strptime(time_str, "%H:%M")
        hour = dt.hour
        next_hour = hour + 1
        return f"{hour:02d}시~{next_hour:02d}시"
    except:
        return "Time Error"

def color_route_style(val):
    val_upper = str(val).upper()
    if any(x in val_upper for x in ['CHN', '중국']): return 'background-color: #d9534f; color: white; font-weight: bold;'
    elif any(x in val_upper for x in ['SEA', '동남아']): return 'background-color: #f0ad4e; color: black; font-weight: bold;'
    elif any(x in val_upper for x in ['JPN', '일본']): return 'background-color: #5bc0de; color: black; font-weight: bold;'
    elif any(x in val_upper for x in ['AME', '미주']): return 'background-color: #0275d8; color: white; font-weight: bold;'
    elif any(x in val_upper for x in ['EUR', '구주', '유럽']): return 'background-color: #5cb85c; color: white; font-weight: bold;'
    elif any(x in val_upper for x in ['OCE', '대양주']): return 'background-color: #5bc0de; color: white; font-weight: bold;'
    elif any(x in val_upper for x in ['CIS', '러시아']): return 'background-color: #777; color: white; font-weight: bold;'
    else: return ''

def apply_scoring(df, min_limit, max_limit, score_weights, time_thresholds):
    if df.empty: return df
    def calculate_row_score(row):
        if row['Status'] != 'Connected': return 0
        conn_min = row['Conn_Min']
        if conn_min <= time_thresholds[0]: return score_weights[0]
        elif conn_min <= time_thresholds[1]: return score_weights[1]
        elif conn_min <= time_thresholds[2]: return score_weights[2]
        elif conn_min <= time_thresholds[3]: return score_weights[3]
        else: return score_weights[4]
    df['Score'] = df.apply(calculate_row_score, axis=1)
    return df

def render_score_settings(key_suffix, min_mct, max_ct):
    with st.sidebar.expander("연결 스코어 설정", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        step = (max_ct - min_mct) / 5
        with c1: s1 = st.number_input("S1", 10, key=f's1{key_suffix}'); t1 = int(min_mct + step)
        with c2: s2 = st.number_input("S2", 8, key=f's2{key_suffix}'); t2 = int(min_mct + step*2)
        with c3: s3 = st.number_input("S3", 6, key=f's3{key_suffix}'); t3 = int(min_mct + step*3)
        with c4: s4 = st.number_input("S4", 4, key=f's4{key_suffix}'); t4 = int(min_mct + step*4)
        with c5: s5 = st.number_input("S5", 2, key=f's5{key_suffix}')
        return [s1, s2, s3, s4, s5], [t1, t2, t3, t4], None

def analyze_connections_flexible(df, min_limit, max_limit, group_a_routes, group_a_ops, group_b_routes, group_b_ops):
    results = []
    def analyze_one_direction(start_routes, start_ops, end_routes, end_ops, direction_label):
        inbound = df[(df['ROUTE'].isin(start_routes)) & (df['OPS'].isin(start_ops)) & (df['구분'] == 'To ICN')].copy()
        outbound = df[(df['ROUTE'].isin(end_routes)) & (df['OPS'].isin(end_ops)) & (df['구분'] == 'From ICN')].copy()
        if inbound.empty or outbound.empty: return []
        merged = pd.merge(inbound.assign(k=1), outbound.assign(k=1), on='k', suffixes=('_IN', '_OUT'))
        local_results = []
        for _, row in merged.iterrows():
            arr = time_to_minutes(row['STA_IN'])
            dep = time_to_minutes(row['STD_OUT'])
            if arr is not None and dep is not None:
                diff = dep - arr
                if diff < 0: diff += 1440 
                status = 'Connected' if min_limit <= diff <= max_limit else 'Disconnect'
                local_results.append({
                    'Direction': direction_label,
                    'Inbound_Route': row['ROUTE_IN'], 'Outbound_Route': row['ROUTE_OUT'],
                    'Inbound_OPS': row['OPS_IN'], 'Outbound_OPS': row['OPS_OUT'],
                    'Inbound_Flt_No': f"{row['OPS_IN']}{row['FLT NO_IN']}", 
                    'Outbound_Flt_No': f"{row['OPS_OUT']}{row['FLT NO_OUT']}",
                    'From': row['ORGN_IN'], 'Via': 'ICN', 'To': row['DEST_OUT'],
                    'Hub_Arr_Time': row['STA_IN'], 'Hub_Dep_Time': row['STD_OUT'],
                    'Conn_Min': diff, 'Status': status
                })
        return local_results

    results.extend(analyze_one_direction(group_a_routes, group_a_ops, group_b_routes, group_b_ops, "Group A -> Group B"))
    is_same_group = set(group_a_routes) == set(group_b_routes) and set(group_a_ops) == set(group_b_ops)
    if not is_same_group:
        results.extend(analyze_one_direction(group_b_routes, group_b_ops, group_a_routes, group_a_ops, "Group B -> Group A"))

    cols = ['Direction', 'Inbound_Route', 'Outbound_Route', 'Inbound_OPS', 'Outbound_OPS', 'Inbound_Flt_No', 'Outbound_Flt_No', 'From', 'Via', 'To', 'Hub_Arr_Time', 'Hub_Dep_Time', 'Conn_Min', 'Status']
    if not results: return pd.DataFrame(columns=cols)
    return pd.DataFrame(results)[cols]

# --- 비교 분석용 함수들 ---
def compare_schedules(df1, df2, min_limit, max_limit, group_a_routes, group_a_ops, group_b_routes, group_b_ops, score_weights, time_thresholds):
    raw_result1 = analyze_connections_flexible(df1, min_limit, max_limit, group_a_routes, group_a_ops, group_b_routes, group_b_ops)
    raw_result2 = analyze_connections_flexible(df2, min_limit, max_limit, group_a_routes, group_a_ops, group_b_routes, group_b_ops)
    
    result1 = apply_scoring(raw_result1, min_limit, max_limit, score_weights, time_thresholds)
    result2 = apply_scoring(raw_result2, min_limit, max_limit, score_weights, time_thresholds)
    
    def create_key(row): return f"{row['Inbound_Flt_No']}_{row['Outbound_Flt_No']}_{row['From']}_{row['To']}"
    
    if not result1.empty: result1['Key'] = result1.apply(create_key, axis=1)
    else: result1['Key'] = []
    if not result2.empty: result2['Key'] = result2.apply(create_key, axis=1)
    else: result2['Key'] = []

    conn1_keys = set(result1[result1['Status'] == 'Connected']['Key'])
    conn2_keys = set(result2[result2['Status'] == 'Connected']['Key'])
    
    common_keys = conn1_keys & conn2_keys
    lost_keys = conn1_keys - conn2_keys
    new_keys = conn2_keys - conn1_keys
    
    lost_connections = result1[result1['Key'].isin(lost_keys) & (result1['Status']=='Connected')].copy()
    new_connections = result2[result2['Key'].isin(new_keys) & (result2['Status']=='Connected')].copy()
    
    time_changes = pd.DataFrame()
    if common_keys:
        c1 = result1[result1['Key'].isin(common_keys)][['Key', 'Conn_Min', 'Score', 'Hub_Arr_Time', 'Hub_Dep_Time']].set_index('Key')
        c2 = result2[result2['Key'].isin(common_keys)][['Key', 'Conn_Min', 'Score', 'Hub_Arr_Time', 'Hub_Dep_Time']].set_index('Key')
        
        merged = c1.join(c2, lsuffix='_1', rsuffix='_2')
        time_changes = merged[(merged['Conn_Min_1'] != merged['Conn_Min_2']) | (merged['Score_1'] != merged['Score_2'])].reset_index()
        meta = result2[['Key', 'Inbound_Flt_No', 'Outbound_Flt_No', 'From', 'To']].drop_duplicates()
        time_changes = pd.merge(time_changes, meta, on='Key', how='left')

    return {
        'result1': result1, 'result2': result2,
        'stats': {
            'total_score_1': result1[result1['Status']=='Connected']['Score'].sum(),
            'total_score_2': result2[result2['Status']=='Connected']['Score'].sum(),
            'total_conn_1': len(conn1_keys), 'total_conn_2': len(conn2_keys),
            'lost_count': len(lost_keys), 'new_count': len(new_keys)
        },
        'lost_connections': lost_connections,
        'new_connections': new_connections,
        'time_changes': time_changes
    }

def compare_flights(df1, df2):
    def create_flight_key(row): return f"{row['OPS']}{row['FLT NO']}_{row['ORGN']}_{row['DEST']}"
    d1 = df1.copy(); d2 = df2.copy()
    d1['Key'] = d1.apply(create_flight_key, axis=1)
    d2['Key'] = d2.apply(create_flight_key, axis=1)
    k1 = set(d1['Key']); k2 = set(d2['Key'])
    removed = d1[d1['Key'].isin(k1 - k2)]
    added = d2[d2['Key'].isin(k2 - k1)]
    common = k1 & k2
    c1 = d1[d1['Key'].isin(common)].set_index('Key')[['STD', 'STA']]
    c2 = d2[d2['Key'].isin(common)].set_index('Key')[['STD', 'STA']]
    m = c1.join(c2, lsuffix='_OLD', rsuffix='_NEW')
    changed = m[(m['STD_OLD'] != m['STD_NEW']) | (m['STA_OLD'] != m['STA_NEW'])].reset_index()
    return {
        'removed': removed, 'added': added, 'time_changed': changed,
        'stats': {'total_1': len(k1), 'total_2': len(k2)}
    }

# [UPDATED] 데이터 변환 함수 (내재화된 Mapping)
def preprocess_export_data(file, target_date):
    try:
        df = pd.read_csv(file)
        target_dt = pd.to_datetime(target_date)
        target_weekday = target_dt.weekday() # 0=Mon
        
        # [INTERNAL MAP] 공항-지역 매핑 내재화
        route_map = {
            # 일본 (Japan)
            'NRT': '일본노선', 'HND': '일본노선', 'KIX': '일본노선', 'FUK': '일본노선', 'NGO': '일본노선',
            'CTS': '일본노선', 'OKA': '일본노선', 'KOJ': '일본노선', 'KMJ': '일본노선', 'HIJ': '일본노선',
            'TAK': '일본노선', 'MYJ': '일본노선', 'FSZ': '일본노선', 'KIJ': '일본노선', 'OKJ': '일본노선',
            'KKJ': '일본노선', 'AOJ': '일본노선', 'AXT': '일본노선', 'HNA': '일본노선', 'KUH': '일본노선',
            'MMB': '일본노선', 'OIT': '일본노선', 'SDJ': '일본노선', 'UBJ': '일본노선', 'UKB': '일본노선',
            'NGS': '일본노선', 'KMQ': '일본노선',
            
            # 중국 (China)
            'PEK': '중국노선', 'PVG': '중국노선', 'SHA': '중국노선', 'CAN': '중국노선', 'HKG': '중국노선',
            'TPE': '중국노선', 'TSN': '중국노선', 'SHE': '중국노선', 'TAO': '중국노선', 'CKG': '중국노선',
            'CTU': '중국노선', 'DLC': '중국노선', 'HGH': '중국노선', 'HRB': '중국노선', 'KMG': '중국노선',
            'NKG': '중국노선', 'SZX': '중국노선', 'TNA': '중국노선', 'WEH': '중국노선', 'XIY': '중국노선',
            'XMN': '중국노선', 'YNJ': '중국노선', 'YNT': '중국노선', 'CGQ': '중국노선', 'CSX': '중국노선',
            'CGO': '중국노선', 'FOC': '중국노선', 'HAK': '중국노선', 'HFE': '중국노선', 'JJN': '중국노선',
            'KWE': '중국노선', 'NNG': '중국노선', 'SYX': '중국노선', 'WUH': '중국노선', 'XNN': '중국노선',
            'DYG': '중국노선', 'MDG': '중국노선', 'MFM': '중국노선', 'RMQ': '중국노선',
            
            # 동남아 (Southeast Asia)
            'BKK': '동남아노선', 'SIN': '동남아노선', 'SGN': '동남아노선', 'HAN': '동남아노선', 'MNL': '동남아노선',
            'CEB': '동남아노선', 'HKT': '동남아노선', 'DAD': '동남아노선', 'KUL': '동남아노선', 'CGK': '동남아노선',
            'DPS': '동남아노선', 'CNX': '동남아노선', 'PNH': '동남아노선', 'RGN': '동남아노선', 'VTE': '동남아노선',
            'CXR': '동남아노선', 'DVO': '동남아노선', 'KBV': '동남아노선', 'KLO': '동남아노선', 'LPQ': '동남아노선',
            'REP': '동남아노선', 'USM': '동남아노선', 'CRK': '동남아노선', 'GUM': '동남아노선', 'PQC': '동남아노선',
            'KTI': '동남아노선', 'KTM': '동남아노선',

            # 미주 (Americas)
            'LAX': '미주노선', 'JFK': '미주노선', 'SFO': '미주노선', 'SEA': '미주노선', 'ATL': '미주노선',
            'ORD': '미주노선', 'LAS': '미주노선', 'HNL': '미주노선', 'YVR': '미주노선', 'YYZ': '미주노선',
            'DFW': '미주노선', 'IAD': '미주노선', 'BOS': '미주노선', 'DTW': '미주노선', 'MSP': '미주노선',
            'SLC': '미주노선',  # 괌은 보통 미주 또는 대양주로 분류 (여기선 미주)
            'ANC': '미주노선', 'MIA': '미주노선', 'IAH': '미주노선',
            
            # 구주/유럽 (Europe)
            'LHR': '구주노선', 'CDG': '구주노선', 'FRA': '구주노선', 'FCO': '구주노선', 'MXP': '구주노선',
            'BCN': '구주노선', 'MAD': '구주노선', 'AMS': '구주노선', 'ZRH': '구주노선', 'IST': '구주노선',
            'PRG': '구주노선', 'BUD': '구주노선', 'VIE': '구주노선', 'MUC': '구주노선', 'LIS': '구주노선',
            'ZAG': '구주노선', 'WAW': '구주노선', 'OSL': '구주노선', 'ARN': '구주노선', 'CPH': '구주노선',
            'HEL': '구주노선', 'SVO': '구주노선', 'LED': '구주노선', 'TLV': '구주노선', # 중동 일부 포함 가능
            'DXB': '구주노선', # 편의상 중동 포함
            
            # 대양주 (Oceania)
            'SYD': '대양주노선', 'BNE': '대양주노선', 'AKL': '대양주노선', 'NAN': '대양주노선', 'ROR': '대양주노선',
            'SPN': '대양주노선',
            
            # CIS
            'UBN': 'CIS노선', 'VVO': 'CIS노선', 'ALA': 'CIS노선', 'TAS': 'CIS노선', 'KHV': 'CIS노선',
            'YKS': 'CIS노선', 'IKT': 'CIS노선'
        }
        
        processed_rows = []
        
        for _, row in df.iterrows():
            try:
                # PERIOD
                period_str = str(row['PERIOD']).strip()
                if '~' in period_str:
                    s_str, e_str = period_str.split('~')
                    start_date = pd.to_datetime(s_str.strip())
                    end_date = pd.to_datetime(e_str.strip())
                    if not (start_date <= target_dt <= end_date): continue
                
                # DAY
                days_ops = str(row['DAY']).strip()
                data_idx = target_weekday + 1
                if str(data_idx) not in days_ops: continue
            except: continue

            # Data Parsing
            raw_flt = str(row['FLT']).strip()
            ops = raw_flt[:2]
            flt_no = raw_flt[2:]
            
            orgn = str(row['DEP']).strip()
            dest = str(row['ARR']).strip()
            std = str(row['STD']).strip()
            sta = str(row['STA']).strip()
            
            # 구분 및 Route 매핑 대상 확인
            if orgn == 'ICN':
                gubun = 'From ICN'
                check_port = dest
            elif dest == 'ICN':
                gubun = 'To ICN'
                check_port = orgn
            else:
                continue
            
            # Route Map 적용
            route = route_map.get(check_port, '기타노선')
            
            processed_rows.append({
                'SEASON': 'S26',
                'FLT NO': flt_no, 'ORGN': orgn, 'DEST': dest,
                'STD': std, 'STA': sta, 'OPS': ops,
                'ROUTE': route, '구분': gubun
            })
            
        return pd.DataFrame(processed_rows)

    except Exception as e:
        st.error(f"데이터 변환 중 오류: {e}")
        return None

# ==============================================================================
# 3. 메인 실행 로직
# ==============================================================================

# [MODE 1] 스케줄 데이터 변환기
if analysis_mode == "스케줄 데이터 변환":
    st.header("Raw 스케줄 데이터 변환")
    st.info("BASE의 WEEKLY SKD 메뉴를 통해 추출한 export.csv 파일을 업로드하고 분석할 날짜를 선택하면 분석가능한 형식으로 변환합니다")
    
    col1, col2 = st.columns(2)
    with col1:
        raw_file = st.file_uploader("원본 파일 업로드 (export.csv)", type="csv")
    with col2:
        target_date = st.date_input("분석할 일자 선택", datetime.today())
        
    if st.button("변환 실행", type="primary"):
        if raw_file:
            with st.spinner("데이터 변환 및 필터링 중..."):
                converted_df = preprocess_export_data(raw_file, target_date)
                
                if converted_df is not None and not converted_df.empty:
                    st.success(f"변환 완료! 총 {len(converted_df)}개의 운항편 추출.")
                    st.dataframe(converted_df.head())
                    
                    st.session_state['converted_data'] = converted_df
                    
                    csv = converted_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("변환된 파일 다운로드", csv, f"Schedule_{target_date}.csv", "text/csv")
                else:
                    st.warning("조건에 맞는 운항 데이터가 없거나 변환 실패.")
        else:
            st.error("파일을 업로드해주세요.")

# [MODE 2] 단일 스케줄 분석
elif analysis_mode == "단일 스케줄 분석":
    st.sidebar.header("분석 설정")
    
    df = None
    use_converted = False
    
    if 'converted_data' in st.session_state:
        st.sidebar.success(f"변환된 데이터 감지됨 ({len(st.session_state['converted_data'])}건)")
        if st.sidebar.checkbox("변환된 데이터 사용하기", value=True):
            df = st.session_state['converted_data']
            use_converted = True
            
    if not use_converted:
        uploaded_file = st.sidebar.file_uploader("분석용 데이터 (CSV)", type="csv")
        if uploaded_file:
            df = load_data(uploaded_file)

    if df is not None:
        if not use_converted:
            st.sidebar.success(f"파일 로드: {len(df)}건")
            
        all_routes = sorted(df['ROUTE'].unique().tolist())
        all_ops = sorted(df['OPS'].unique().tolist())
        
        st.sidebar.markdown("---")
        routes_a = st.sidebar.multiselect("그룹 A 노선", all_routes, default=[all_routes[0]] if all_routes else None, key='ra')
        ops_a = st.sidebar.multiselect("그룹 A 항공사", all_ops, default=all_ops, key='oa')
        
        st.sidebar.markdown("⬇️ ⬆️")
        
        routes_b = st.sidebar.multiselect("그룹 B 노선", all_routes, default=[all_routes[1]] if len(all_routes)>1 else all_routes, key='rb')
        ops_b = st.sidebar.multiselect("그룹 B 항공사", all_ops, default=all_ops, key='ob')
        
        st.sidebar.markdown("---")
        min_mct = st.sidebar.number_input("Min CT (분)", 0, 300, 60, 5)
        max_ct = st.sidebar.number_input("Max CT (분)", 60, 2880, 300, 60)
        
        score_weights, time_thresholds, _ = render_score_settings("single", min_mct, max_ct)
        
        if st.button("분석 시작", type="primary"):
            if not routes_a or not routes_b:
                st.error("그룹 노선을 선택해주세요.")
            else:
                with st.spinner("분석 중..."):
                    raw_df = analyze_connections_flexible(df, min_mct, max_ct, routes_a, ops_a, routes_b, ops_b)
                    result_df = apply_scoring(raw_df, min_mct, max_ct, score_weights, time_thresholds)
                    st.session_state['analysis_result'] = result_df
                    st.session_state['analysis_done'] = True
                    st.session_state['group_names'] = (", ".join(routes_a), ", ".join(routes_b))
                    st.session_state['source_df'] = df

        if st.session_state.get('analysis_done'):
            result_df = st.session_state['analysis_result']
            source_df = st.session_state.get('source_df', df)
            g_name_a, g_name_b = st.session_state.get('group_names', ("A", "B"))
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["결과 요약", "상세 리스트", "공항별 심층 분석", "허브 스케줄", "Bank 시각화"])
            
            with tab1:
                st.info(f"분석 기준: [{g_name_a}] ↔ [{g_name_b}]")
                if not result_df.empty:
                    m1, m2 = st.columns(2)
                    m1.metric("총 연결 편수", f"{len(result_df[result_df['Status']=='Connected']):,}편")
                    m2.metric("평균 스코어", f"{result_df[result_df['Status']=='Connected']['Score'].mean():.1f}점")
                    st.dataframe(result_df.groupby(['Inbound_Route', 'Outbound_Route', 'Status']).size().unstack(fill_value=0), use_container_width=True)
                else:
                    st.warning("결과가 없습니다.")

            with tab2:
                if not result_df.empty:
                    st.dataframe(result_df[result_df['Status']=='Connected'], use_container_width=True)
            
            with tab3: # 공항별 분석
                if result_df.empty:
                     st.warning("데이터가 없습니다.")
                else:
                    st.markdown("### 공항 기준 연결성 분석")
                    src_a = result_df[result_df['Direction'] == 'Group A -> Group B']['From'].unique()
                    dst_a = result_df[result_df['Direction'] == 'Group B -> Group A']['To'].unique()
                    candidates = set(src_a) | set(dst_a)
                    if 'ICN' in candidates: candidates.remove('ICN')
                    airport_list = sorted(list(candidates))
                    
                    if not airport_list:
                        st.info("차트를 그릴 수 있는 공항 데이터가 없습니다.")
                    else:
                        st.markdown(f"**그룹 A ({g_name_a}) 소속 공항 선택**")
                        selected_airport = st.selectbox("📍 공항 선택", airport_list)
                        connected_data = result_df[result_df['Status']=='Connected']
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            out_df = connected_data[(connected_data['Direction'] == 'Group A -> Group B') & (connected_data['From'] == selected_airport)]
                            if not out_df.empty:
                                chart = alt.Chart(out_df).mark_circle(size=150).encode(
                                    x='To', y='Conn_Min', color='Inbound_Flt_No', 
                                    tooltip=['To', 'Conn_Min', 'Inbound_Flt_No', 'Outbound_Flt_No', 'Hub_Arr_Time', 'Hub_Dep_Time']
                                ).properties(height=500, title=f"{selected_airport} 도착 -> ICN 연결").interactive()
                                st.altair_chart(chart, use_container_width=True)
                            else: st.info("데이터 없음")
                        with c2:
                            in_df = connected_data[(connected_data['Direction'] == 'Group B -> Group A') & (connected_data['To'] == selected_airport)]
                            if not in_df.empty:
                                chart = alt.Chart(in_df).mark_circle(size=150).encode(
                                    x='From', y='Conn_Min', color='Outbound_Flt_No', 
                                    tooltip=['From', 'Conn_Min', 'Outbound_Flt_No', 'Inbound_Flt_No', 'Hub_Arr_Time', 'Hub_Dep_Time']
                                ).properties(height=500, title=f"ICN 출발 -> {selected_airport} 도착").interactive()
                                st.altair_chart(chart, use_container_width=True)
                            else: st.info("데이터 없음")

            with tab4: # 허브 스케줄
                st.markdown("### ICN 허브 스케줄 모니터링")
                st.caption("도착/출발 항공편을 1시간 단위로 분류하여 노선별 색상 코드로 시각화합니다.")
                arr_raw = source_df[source_df['구분'] == 'To ICN'].copy()
                dep_raw = source_df[source_df['구분'] == 'From ICN'].copy()
                arr_raw['시간대'] = arr_raw['STA'].apply(get_time_slot)
                dep_raw['시간대'] = dep_raw['STD'].apply(get_time_slot)
                arr_raw['Sort_Key'] = arr_raw['STA'].apply(time_to_minutes)
                dep_raw['Sort_Key'] = dep_raw['STD'].apply(time_to_minutes)
                arr_raw = arr_raw.sort_values(by='Sort_Key', ascending=True)
                dep_raw = dep_raw.sort_values(by='Sort_Key', ascending=True)
                cols_arr = ['시간대', 'STA', 'ROUTE', 'ORGN', 'OPS', 'FLT NO']
                cols_dep = ['시간대', 'STD', 'ROUTE', 'DEST', 'OPS', 'FLT NO']
                styled_arr = arr_raw[cols_arr].style.map(color_route_style, subset=['ROUTE'])
                styled_dep = dep_raw[cols_dep].style.map(color_route_style, subset=['ROUTE'])
                col_arr, col_dep = st.columns(2)
                with col_arr:
                    st.subheader("🛬 ICN 도착 (Arrival)")
                    st.dataframe(styled_arr, use_container_width=True, height=800, hide_index=True)
                with col_dep:
                    st.subheader("🛫 ICN 출발 (Departure)")
                    st.dataframe(styled_dep, use_container_width=True, height=800, hide_index=True)

            with tab5: # Interactive Bank
                st.markdown("### Connection Bank (Interactive)")
                st.caption("왼쪽(Inbound)을 클릭하면 연결 가능한 오른쪽(Outbound) 편이 강조됩니다.")

                if 'selected_inbound_flt' not in st.session_state:
                    st.session_state['selected_inbound_flt'] = None

                target_df = result_df[(result_df['Status'] == 'Connected') & (result_df['Direction'] == 'Group A -> Group B')].copy()

                if target_df.empty:
                    st.warning("조건에 맞는 연결 데이터가 없습니다.")
                else:
                    in_cols = ['Inbound_Flt_No', 'Inbound_OPS', 'Inbound_Route', 'From', 'Hub_Arr_Time']
                    df_in = target_df[in_cols].drop_duplicates()
                    df_in.columns = ['FLT', 'OPS', 'ROUTE', 'PORT', 'TIME']
                    df_in['Time_Min'] = df_in['TIME'].apply(time_to_minutes)
                    df_in['Hour'] = (df_in['Time_Min'] // 60) % 24
                    
                    out_cols = ['Outbound_Flt_No', 'Outbound_OPS', 'Outbound_Route', 'To', 'Hub_Dep_Time']
                    df_out = target_df[out_cols].drop_duplicates()
                    df_out.columns = ['FLT', 'OPS', 'ROUTE', 'PORT', 'TIME']
                    df_out['Time_Min'] = df_out['TIME'].apply(time_to_minutes)
                    df_out['Hour'] = (df_out['Time_Min'] // 60) % 24

                    df_in = df_in.sort_values(by=['Time_Min', 'ROUTE'])
                    df_out = df_out.sort_values(by=['Time_Min', 'ROUTE'])

                    def get_route_color_hex(route_val):
                        val_upper = str(route_val).upper()
                        if any(x in val_upper for x in ['CHN', '중국']): return '#d9534f'
                        elif any(x in val_upper for x in ['SEA', '동남아']): return '#f0ad4e'
                        elif any(x in val_upper for x in ['JPN', '일본']): return '#5bc0de'
                        elif any(x in val_upper for x in ['AME', '미주']): return '#0275d8'
                        elif any(x in val_upper for x in ['EUR', '구주']): return '#5cb85c'
                        return '#777777'

                    def create_outbound_card(row, is_highlighted, is_dimmed):
                        bg_color = get_route_color_hex(row['ROUTE'])
                        opacity = "0.2" if is_dimmed else "1.0"
                        box_shadow = "0px 0px 8px 2px #FFD700" if is_highlighted else "1px 1px 3px rgba(0,0,0,0.1)"
                        border_style = f"4px solid {bg_color}"
                        html = f"""
                        <div style="opacity:{opacity}; border-left:{border_style}; padding:10px; margin-bottom:8px; background:white; box-shadow:{box_shadow}; transition:all 0.3s ease; border-radius:4px;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-weight:bold; color:#333; font-size:1.1em;">{row['TIME']}</span>
                                <span style="background-color:{bg_color}; color:white; padding:2px 6px; border-radius:3px; font-size:0.7em;">{row['ROUTE']}</span>
                            </div>
                            <div style="margin-top:4px; display:flex; justify-content:space-between; color:#555;">
                                <span>{row['FLT']}</span>
                                <span style="font-weight:bold;">{row['PORT']}</span>
                            </div>
                        </div>"""
                        return html

                    connected_outbounds = []
                    if st.session_state['selected_inbound_flt']:
                        connected_outbounds = target_df[target_df['Inbound_Flt_No'] == st.session_state['selected_inbound_flt']]['Outbound_Flt_No'].tolist()

                    for hour in range(24):
                        in_group = df_in[df_in['Hour'] == hour]
                        out_group = df_out[df_out['Hour'] == hour]

                        if not in_group.empty or not out_group.empty:
                            st.markdown(f"<div style='background:#f0f2f6; padding:5px; margin:10px 0; font-weight:bold; text-align:center; border-radius:5px;'>{hour:02d}:00 - {hour+1:02d}:00</div>", unsafe_allow_html=True)
                            c1, c2 = st.columns(2)
                            with c1:
                                for _, row in in_group.iterrows():
                                    flt = row['FLT']
                                    icon = "🔵" if st.session_state['selected_inbound_flt'] == flt else "⚪"
                                    if st.button(f"{icon} [{row['TIME']}] {flt} ({row['PORT']})", key=f"btn_{flt}", use_container_width=True):
                                        st.session_state['selected_inbound_flt'] = flt
                                        st.rerun()
                            with c2:
                                for _, row in out_group.iterrows():
                                    flt_out = row['FLT']
                                    is_highlight = (st.session_state['selected_inbound_flt'] and flt_out in connected_outbounds)
                                    is_dim = (st.session_state['selected_inbound_flt'] and not is_highlight)
                                    st.markdown(create_outbound_card(row, is_highlight, is_dim), unsafe_allow_html=True)
                    
                    if st.session_state['selected_inbound_flt']:
                        if st.button("🔄 선택 초기화"):
                            st.session_state['selected_inbound_flt'] = None
                            st.rerun()

# [MODE 3] 두 스케줄 비교 분석
elif analysis_mode == "두 스케줄 비교 분석":
    st.sidebar.header("⚙️ 비교 분석 설정")
    f1 = st.sidebar.file_uploader("📂 스케줄 1 (Before)", type="csv", key="f1")
    f2 = st.sidebar.file_uploader("📂 스케줄 2 (After)", type="csv", key="f2")
    
    if f1 and f2:
        try:
            df1 = load_data(f1)
            df2 = load_data(f2)
            
            all_routes = sorted(set(df1['ROUTE'].unique().tolist() + df2['ROUTE'].unique().tolist()))
            all_ops = sorted(set(df1['OPS'].unique().tolist() + df2['OPS'].unique().tolist()))
            
            st.sidebar.markdown("---")
            routes_a = st.sidebar.multiselect("그룹 A 노선", all_routes, key='cmp_ra')
            ops_a = st.sidebar.multiselect("그룹 A 항공사", all_ops, default=all_ops, key='cmp_oa')
            routes_b = st.sidebar.multiselect("그룹 B 노선", all_routes, key='cmp_rb')
            ops_b = st.sidebar.multiselect("그룹 B 항공사", all_ops, default=all_ops, key='cmp_ob')
            
            min_mct = st.sidebar.number_input("Min CT", 0, 300, 60, 5, key='cmp_min')
            max_ct = st.sidebar.number_input("Max CT", 60, 2880, 300, 60, key='cmp_max')
            score_weights_cmp, time_thresholds_cmp, _ = render_score_settings("cmp", min_mct, max_ct)
            
            if st.button("🔍 비교 분석 시작", type="primary"):
                 if routes_a and routes_b:
                    with st.spinner("비교 분석 중..."):
                        conn_cmp = compare_schedules(df1, df2, min_mct, max_ct, routes_a, ops_a, routes_b, ops_b, score_weights_cmp, time_thresholds_cmp)
                        flt_cmp = compare_flights(df1, df2)
                        st.session_state['conn_comparison'] = conn_cmp
                        st.session_state['flight_comparison'] = flt_cmp
                        st.session_state['comparison_done'] = True
                        st.session_state['cmp_group_names'] = (", ".join(routes_a), ", ".join(routes_b))
                 else:
                     st.error("그룹을 선택해주세요.")
            
            if st.session_state.get('comparison_done'):
                conn_cmp = st.session_state['conn_comparison']
                flt_cmp = st.session_state['flight_comparison']
                g_name_a, g_name_b = st.session_state.get('cmp_group_names', ("A", "B"))
                
                t1, t2, t3, t4 = st.tabs(["📊 비교 요약", "✈️ 항공편 변경", "🔗 연결 변경", "⏱️ 시간/스코어 변경"])
                
                with t1:
                    st.info(f"**분석 기준**: [{g_name_a}] ↔ [{g_name_b}]")
                    sc_col1, sc_col2, sc_col3 = st.columns(3)
                    with sc_col1: st.metric("스케줄 1 총점", f"{conn_cmp['stats']['total_score_1']:,.0f}점")
                    with sc_col2: st.metric("스케줄 2 총점", f"{conn_cmp['stats']['total_score_2']:,.0f}점")
                    with sc_col3: 
                        diff = conn_cmp['stats']['total_score_2'] - conn_cmp['stats']['total_score_1']
                        st.metric("점수 차이", f"{diff:+,.0f}점", delta=diff)
                    st.markdown("---")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("#### ✈️ 항공편 변경")
                        st.metric("총 항공편 차이", flt_cmp['stats']['total_2'] - flt_cmp['stats']['total_1'])
                    with c2:
                        st.markdown("#### 🔗 연결 변경")
                        st.metric("총 연결 편수 차이", conn_cmp['stats']['total_conn_2'] - conn_cmp['stats']['total_conn_1'])

                with t2:
                    if not flt_cmp['time_changed'].empty:
                        st.markdown("#### 🕒 시간 변경된 항공편")
                        st.dataframe(flt_cmp['time_changed'])
                    col_rem, col_add = st.columns(2)
                    with col_rem:
                        st.markdown("#### ❌ 삭제된 항공편")
                        if not flt_cmp['removed'].empty: st.dataframe(flt_cmp['removed'])
                    with col_add:
                        st.markdown("#### 🆕 신규 항공편")
                        if not flt_cmp['added'].empty: st.dataframe(flt_cmp['added'])
                
                with t3:
                    st.markdown("**❌ 사라진 연결**")
                    st.dataframe(conn_cmp['lost_connections'])
                    st.markdown("**🆕 새로운 연결**")
                    st.dataframe(conn_cmp['new_connections'])
                
                with t4:
                    st.markdown("**⏱️ 연결 시간/스코어 변경 상세**")
                    st.dataframe(conn_cmp['time_changes'])

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            import traceback
            st.text(traceback.format_exc())
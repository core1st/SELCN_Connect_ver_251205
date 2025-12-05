import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from datetime import datetime

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
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("연결 스케줄 확인 앱 VER.2.6 (Complete Integrated)")

# --- 모드 선택 ---
analysis_mode = st.radio(
    "분석 모드 선택",
    ["단일 스케줄 분석", "두 스케줄 비교 분석"],
    horizontal=True
)

# --- [NOTICE] 데이터 작성 가이드 ---
with st.expander("[필독] 데이터 파일(CSV) 작성 양식 가이드", expanded=False):
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
# 2. 공통 함수 정의 (데이터 로드, 처리, 스타일링)
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
            if not all(col in df.columns for col in required):
                continue
            
            for col in ['구분', 'FLT NO', 'ROUTE', 'OPS', 'ORGN', 'DEST']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
            return df
        except:
            continue
    raise ValueError("파일을 읽을 수 없습니다. 인코딩 문제이거나 필수 컬럼이 누락되었습니다.")

def time_to_minutes(t_str):
    try:
        h, m = map(int, t_str.split(':'))
        return h * 60 + m
    except:
        return None

# [NEW] 시간대 그룹화 함수
def get_time_slot(time_str):
    """HH:MM 문자열을 받아서 HH시~HH+1시 문자열 반환"""
    try:
        dt = datetime.strptime(time_str, "%H:%M")
        hour = dt.hour
        next_hour = hour + 1
        return f"{hour:02d}시~{next_hour:02d}시"
    except:
        return "Time Error"

# [NEW] 노선별 색상 스타일링 함수
def color_route_style(val):
    """ROUTE 값에 따라 배경색과 글자색 CSS 반환"""
    val_upper = str(val).upper()
    if any(x in val_upper for x in ['CHN', '중국']):
        return 'background-color: #d9534f; color: white; font-weight: bold;'
    elif any(x in val_upper for x in ['SEA', '동남아']):
        return 'background-color: #f0ad4e; color: black; font-weight: bold;'
    elif any(x in val_upper for x in ['JPN', '일본']):
        return 'background-color: #5bc0de; color: black; font-weight: bold;'
    elif any(x in val_upper for x in ['AME', '미주']):
        return 'background-color: #0275d8; color: white; font-weight: bold;'
    elif any(x in val_upper for x in ['EUR', '구주', '유럽']):
        return 'background-color: #5cb85c; color: white; font-weight: bold;'
    else:
        return ''

# 스코어 계산 함수
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

# UI 헬퍼: 스코어 설정
def render_score_settings(key_suffix, min_mct, max_ct):
    with st.sidebar.expander("연결 스코어 설정", expanded=False):
        st.markdown("**단계별 부여 점수 및 시간 기준**")
        c1, c2, c3, c4, c5 = st.columns(5)
        step = (max_ct - min_mct) / 5
        
        with c1:
            st.markdown("Tier 1")
            s1 = st.number_input("점수", value=10, key=f's1_{key_suffix}')
            t1 = st.number_input("분", value=int(min_mct + step), key=f't1_{key_suffix}')
        with c2:
            st.markdown("Tier 2")
            s2 = st.number_input("점수", value=8, key=f's2_{key_suffix}')
            t2 = st.number_input("분", value=int(min_mct + step*2), key=f't2_{key_suffix}')
        with c3:
            st.markdown("Tier 3")
            s3 = st.number_input("점수", value=6, key=f's3_{key_suffix}')
            t3 = st.number_input("분", value=int(min_mct + step*3), key=f't3_{key_suffix}')
        with c4:
            st.markdown("Tier 4")
            s4 = st.number_input("점수", value=4, key=f's4_{key_suffix}')
            t4 = st.number_input("분", value=int(min_mct + step*4), key=f't4_{key_suffix}')
        with c5:
            st.markdown("Tier 5")
            s5 = st.number_input("점수", value=2, key=f's5_{key_suffix}')
            st.caption(f"~ {max_ct}분")
            
        score_weights = [s1, s2, s3, s4, s5]
        time_thresholds = [t1, t2, t3, t4]
        return score_weights, time_thresholds, (s1, t1, s5)

# ==============================================================================
# 3. 분석 핵심 로직 (단일 & 비교 공용)
# ==============================================================================

def analyze_connections_flexible(df, min_limit, max_limit, 
                               group_a_routes, group_a_ops, 
                               group_b_routes, group_b_ops):
    results = []
    
    def analyze_one_direction(start_routes, start_ops, end_routes, end_ops, direction_label):
        inbound = df[
            (df['ROUTE'].isin(start_routes)) & 
            (df['OPS'].isin(start_ops)) & 
            (df['구분'] == 'To ICN')
        ].copy()
        
        outbound = df[
            (df['ROUTE'].isin(end_routes)) & 
            (df['OPS'].isin(end_ops)) & 
            (df['구분'] == 'From ICN')
        ].copy()

        if inbound.empty or outbound.empty:
            return []

        local_results = []
        # Cross Join을 위한 임시 키
        merged = pd.merge(inbound.assign(k=1), outbound.assign(k=1), on='k', suffixes=('_IN', '_OUT'))
        
        for _, row in merged.iterrows():
            arr = time_to_minutes(row['STA_IN'])
            dep = time_to_minutes(row['STD_OUT'])
            
            if arr is not None and dep is not None:
                diff = dep - arr
                if diff < 0: diff += 1440 # 익일 연결 처리
                
                status = 'Connected' if min_limit <= diff <= max_limit else 'Disconnect'
                
                flt_in = f"{row['OPS_IN']}{row['FLT NO_IN']}"
                flt_out = f"{row['OPS_OUT']}{row['FLT NO_OUT']}"

                local_results.append({
                    'Direction': direction_label,
                    'Inbound_Route': row['ROUTE_IN'],
                    'Outbound_Route': row['ROUTE_OUT'],
                    'Inbound_OPS': row['OPS_IN'], 'Outbound_OPS': row['OPS_OUT'],
                    'Inbound_Flt_No': flt_in, 'Outbound_Flt_No': flt_out,
                    'From': row['ORGN_IN'], 
                    'Via': 'ICN', 
                    'To': row['DEST_OUT'],
                    'Hub_Arr_Time': row['STA_IN'], 'Hub_Dep_Time': row['STD_OUT'],
                    'Arr_Min': arr, 'Dep_Min': dep,
                    'Arr_Hour': arr / 60.0, 
                    'Dep_Hour': dep / 60.0,
                    'Conn_Min': diff, 'Status': status
                })
        return local_results

    # A -> B
    results.extend(analyze_one_direction(group_a_routes, group_a_ops, group_b_routes, group_b_ops, "Group A -> Group B"))
    
    # B -> A (그룹이 다를 경우에만 수행)
    is_same_group = set(group_a_routes) == set(group_b_routes) and set(group_a_ops) == set(group_b_ops)
    if not is_same_group:
        results.extend(analyze_one_direction(group_b_routes, group_b_ops, group_a_routes, group_a_ops, "Group B -> Group A"))

    cols = ['Direction', 'Inbound_Route', 'Outbound_Route', 'Inbound_OPS', 'Outbound_OPS', 'Inbound_Flt_No', 'Outbound_Flt_No', 'From', 'Via', 'To', 'Hub_Arr_Time', 'Hub_Dep_Time', 'Arr_Min', 'Dep_Min', 'Arr_Hour', 'Dep_Hour', 'Conn_Min', 'Status']
    if not results: return pd.DataFrame(columns=cols)
    return pd.DataFrame(results)[cols]


def compare_schedules(df1, df2, min_limit, max_limit, 
                      group_a_routes, group_a_ops, 
                      group_b_routes, group_b_ops,
                      score_weights, time_thresholds):
    """두 스케줄의 연결 분석 결과를 비교 (복구됨)"""
    
    # 각 스케줄 분석 실행
    raw_result1 = analyze_connections_flexible(df1, min_limit, max_limit, group_a_routes, group_a_ops, group_b_routes, group_b_ops)
    raw_result2 = analyze_connections_flexible(df2, min_limit, max_limit, group_a_routes, group_a_ops, group_b_routes, group_b_ops)
    
    # 스코어 적용
    result1 = apply_scoring(raw_result1, min_limit, max_limit, score_weights, time_thresholds)
    result2 = apply_scoring(raw_result2, min_limit, max_limit, score_weights, time_thresholds)
    
    # 연결 쌍 식별 키 생성
    def create_connection_key(row):
        return f"{row['Inbound_Flt_No']}_{row['Outbound_Flt_No']}_{row['From']}_{row['To']}"
    
    if not result1.empty: result1['Connection_Key'] = result1.apply(create_connection_key, axis=1)
    else: result1['Connection_Key'] = []
        
    if not result2.empty: result2['Connection_Key'] = result2.apply(create_connection_key, axis=1)
    else: result2['Connection_Key'] = []
    
    # Connected 상태인 키만 추출
    conn1 = set(result1[result1['Status'] == 'Connected']['Connection_Key'].tolist())
    conn2 = set(result2[result2['Status'] == 'Connected']['Connection_Key'].tolist())
    
    # 차이 분석
    only_in_1 = conn1 - conn2
    only_in_2 = conn2 - conn1
    common = conn1 & conn2
    
    # 1. 사라진 연결
    lost_connections = result1[
        (result1['Connection_Key'].isin(only_in_1)) & 
        (result1['Status'] == 'Connected')
    ].copy()
    
    # 2. 새로 생긴 연결
    new_connections = result2[
        (result2['Connection_Key'].isin(only_in_2)) & 
        (result2['Status'] == 'Connected')
    ].copy()
    
    # 3. 시간/점수 변화 (공통 연결)
    common_df1 = result1[
        (result1['Connection_Key'].isin(common)) & 
        (result1['Status'] == 'Connected')
    ][['Connection_Key', 'Conn_Min', 'Hub_Arr_Time', 'Hub_Dep_Time', 'Score']].copy()
    common_df1.columns = ['Connection_Key', 'Conn_Min_1', 'Arr_Time_1', 'Dep_Time_1', 'Score_1']
    
    common_df2 = result2[
        (result2['Connection_Key'].isin(common)) & 
        (result2['Status'] == 'Connected')
    ][['Connection_Key', 'Conn_Min', 'Hub_Arr_Time', 'Hub_Dep_Time', 'Score']].copy()
    common_df2.columns = ['Connection_Key', 'Conn_Min_2', 'Arr_Time_2', 'Dep_Time_2', 'Score_2']
    
    time_changes = pd.merge(common_df1, common_df2, on='Connection_Key')
    time_changes['Time_Diff'] = time_changes['Conn_Min_2'] - time_changes['Conn_Min_1']
    time_changes['Score_Diff'] = time_changes['Score_2'] - time_changes['Score_1']
    
    # 변화가 있는 것만 필터링
    time_changes = time_changes[time_changes['Time_Diff'] != 0].copy()
    
    # UI 표시를 위해 메타 정보(From, To 등) 추가 병합
    if not time_changes.empty:
        meta = result2[['Connection_Key', 'Inbound_Flt_No', 'Outbound_Flt_No', 'From', 'To']].drop_duplicates()
        time_changes = pd.merge(time_changes, meta, on='Connection_Key', how='left')
    
    return {
        'result1': result1,
        'result2': result2,
        'lost_connections': lost_connections,
        'new_connections': new_connections,
        'time_changes': time_changes,
        'stats': {
            'total_conn_1': len(conn1),
            'total_conn_2': len(conn2),
            'total_score_1': result1[result1['Status']=='Connected']['Score'].sum(),
            'total_score_2': result2[result2['Status']=='Connected']['Score'].sum(),
            'lost': len(only_in_1),
            'new': len(only_in_2),
            'common': len(common),
            'time_changed': len(time_changes)
        }
    }


def compare_flights(df1, df2):
    """두 스케줄의 항공편 자체를 비교 (복구됨)"""
    def create_flight_key(row):
        return f"{row['OPS']}{row['FLT NO']}_{row['ORGN']}_{row['DEST']}"
    
    df1_copy = df1.copy()
    df2_copy = df2.copy()
    df1_copy['Flight_Key'] = df1_copy.apply(create_flight_key, axis=1)
    df2_copy['Flight_Key'] = df2_copy.apply(create_flight_key, axis=1)
    
    flights1 = set(df1_copy['Flight_Key'].tolist())
    flights2 = set(df2_copy['Flight_Key'].tolist())
    
    only_in_1 = flights1 - flights2
    only_in_2 = flights2 - flights1
    common = flights1 & flights2
    
    removed_flights = df1_copy[df1_copy['Flight_Key'].isin(only_in_1)].copy()
    added_flights = df2_copy[df2_copy['Flight_Key'].isin(only_in_2)].copy()
    
    # 시간 변경 확인
    common_df1 = df1_copy[df1_copy['Flight_Key'].isin(common)][['Flight_Key', 'STD', 'STA', 'OPS', 'FLT NO', 'ORGN', 'DEST', 'ROUTE', '구분']].copy()
    common_df2 = df2_copy[df2_copy['Flight_Key'].isin(common)][['Flight_Key', 'STD', 'STA']].copy()
    
    merged = pd.merge(common_df1, common_df2, on='Flight_Key', suffixes=('_OLD', '_NEW'))
    time_changed = merged[(merged['STD_OLD'] != merged['STD_NEW']) | (merged['STA_OLD'] != merged['STA_NEW'])].copy()
    
    return {
        'removed': removed_flights,
        'added': added_flights,
        'time_changed': time_changed,
        'stats': {
            'total_1': len(flights1),
            'total_2': len(flights2),
            'removed': len(only_in_1),
            'added': len(only_in_2),
            'time_changed': len(time_changed)
        }
    }

# ==============================================================================
# 4. 메인 실행 로직: [단일 분석 모드]
# ==============================================================================
if analysis_mode == "단일 스케줄 분석":
    st.sidebar.header("⚙️ 분석 설정")
    uploaded_file = st.sidebar.file_uploader("📂 데이터 파일 (CSV)", type="csv")

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.sidebar.success(f"✅ 파일 로드: {len(df)}건")
            
            all_routes = sorted(df['ROUTE'].unique().tolist())
            all_ops = sorted(df['OPS'].unique().tolist())
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("📌 노선 그룹 매칭")
            
            default_route_a = [all_routes[0]] if all_routes else None
            if "미주노선" in all_routes: default_route_a = ["미주노선"]
            routes_a = st.sidebar.multiselect("그룹 A 노선 선택", all_routes, default=default_route_a, key='ra')
            ops_a = st.sidebar.multiselect("그룹 A 항공사 선택", all_ops, default=all_ops, key='oa')
            
            st.sidebar.markdown("⬇️ ⬆️")
            
            default_route_b = [all_routes[1]] if len(all_routes) > 1 else all_routes
            if "동남아노선" in all_routes and "미주노선" in all_routes: default_route_b = ["동남아노선"]
            routes_b = st.sidebar.multiselect("그룹 B 노선 선택", all_routes, default=default_route_b, key='rb')
            ops_b = st.sidebar.multiselect("그룹 B 항공사 선택", all_ops, default=all_ops, key='ob')
            
            st.sidebar.markdown("---")
            min_mct = st.sidebar.number_input("Min CT (분)", 0, 300, 60, 5)
            max_ct = st.sidebar.number_input("Max CT (분)", 60, 2880, 300, 60)
            
            score_weights, time_thresholds, display_info = render_score_settings("single", min_mct, max_ct)
            
            if st.button("🚀 분석 시작", type="primary"):
                if not routes_a or not routes_b:
                    st.error("그룹 노선을 선택해주세요.")
                else:
                    with st.spinner("분석 중..."):
                        raw_df = analyze_connections_flexible(df, min_mct, max_ct, routes_a, ops_a, routes_b, ops_b)
                        result_df = apply_scoring(raw_df, min_mct, max_ct, score_weights, time_thresholds)
                        
                        st.session_state['analysis_result'] = result_df
                        st.session_state['analysis_done'] = True
                        st.session_state['group_names'] = (", ".join(routes_a), ", ".join(routes_b))
                        st.session_state['score_info'] = display_info 
                        st.session_state['source_df'] = df 

            if 'analysis_done' in st.session_state and st.session_state['analysis_done']:
                result_df = st.session_state['analysis_result']
                source_df = st.session_state.get('source_df', df) 
                g_name_a, g_name_b = st.session_state.get('group_names', ("A", "B"))
                
                # 탭 4개 (허브 모니터링 포함)
                tab1, tab2, tab3, tab4 = st.tabs(["📊 결과 요약", "📋 상세 리스트", "✈️ 공항별 심층 분석", "🕒 허브 스케줄 모니터링"])
                
                with tab1:
                    st.info(f" **분석 기준**: [{g_name_a}] ↔ [{g_name_b}]")
                    if result_df.empty:
                        st.warning("조건에 맞는 연결편이 없습니다.")
                    else:
                        st.markdown("#### 스케줄 가치 평가 (Scoring)")
                        total_score = result_df[result_df['Status']=='Connected']['Score'].sum()
                        avg_score = result_df[result_df['Status']=='Connected']['Score'].mean()
                        conn_count = len(result_df[result_df['Status']=='Connected'])
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("총 연결 편수", f"{conn_count:,}편")
                        m2.metric("총 스케줄 점수", f"{total_score:,.0f}점")
                        m3.metric("평균 연결 품질", f"{avg_score:.1f}점 / 10점")
                        
                        st.markdown("---")
                        st.markdown("#### 1️⃣ 노선/항공사별 통합 연결 상세")
                        combined_summary = result_df.groupby(['Inbound_Route', 'Inbound_OPS', 'Outbound_Route', 'Outbound_OPS', 'Status']).size().unstack(fill_value=0)
                        if 'Connected' not in combined_summary.columns: combined_summary['Connected'] = 0
                        combined_summary['Total'] = combined_summary.sum(axis=1)
                        combined_summary = combined_summary.sort_values(by='Connected', ascending=False)
                        st.dataframe(combined_summary, use_container_width=True)

                with tab2:
                    if result_df.empty:
                        st.warning("데이터가 없습니다.")
                    else:
                        st.markdown("#### 상세 연결 리스트")
                        status_filter = st.multiselect("상태 필터", ['Connected', 'Disconnect'], default=['Connected'], key='sf')
                        view_df = result_df[result_df['Status'].isin(status_filter)].sort_values(['Direction', 'Conn_Min'])
                        st.dataframe(view_df, use_container_width=True, hide_index=True)
                        csv = view_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button("💾 CSV 다운로드", csv, "connection_analysis.csv", "text/csv")

                with tab3:
                    if result_df.empty:
                         st.warning("데이터가 없습니다.")
                    else:
                        st.markdown("### 🏙️ 공항 기준 연결성 분석")
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
                                    chart = alt.Chart(out_df).mark_circle(size=100).encode(
                                        x='To', y='Conn_Min', color='Inbound_Flt_No', tooltip=['To', 'Conn_Min', 'Inbound_Flt_No']
                                    ).properties(height=300, title=f"{selected_airport} 도착 -> ICN 연결").interactive()
                                    st.altair_chart(chart, use_container_width=True)
                                else: st.info("데이터 없음")
                            with c2:
                                in_df = connected_data[(connected_data['Direction'] == 'Group B -> Group A') & (connected_data['To'] == selected_airport)]
                                if not in_df.empty:
                                    chart = alt.Chart(in_df).mark_circle(size=100).encode(
                                        x='From', y='Conn_Min', color='Outbound_Flt_No', tooltip=['From', 'Conn_Min', 'Outbound_Flt_No']
                                    ).properties(height=300, title=f"ICN 출발 -> {selected_airport} 도착").interactive()
                                    st.altair_chart(chart, use_container_width=True)
                                else: st.info("데이터 없음")

                # [NEW] Tab 4: Hub Schedule Monitor
                with tab4:
                    st.markdown("### 🕒 ICN 허브 스케줄 모니터링")
                    st.caption("도착/출발 항공편을 1시간 단위로 분류하여 노선별 색상 코드로 시각화합니다.")
                    
                    # 1. 데이터 분리 (도착/출발) 및 시간대 생성
                    arr_raw = source_df[source_df['구분'] == 'To ICN'].copy()
                    dep_raw = source_df[source_df['구분'] == 'From ICN'].copy()
                    
                    arr_raw['시간대'] = arr_raw['STA'].apply(get_time_slot)
                    dep_raw['시간대'] = dep_raw['STD'].apply(get_time_slot)
                    
                    # 2. 정렬 (시간순 - 분 단위 변환 값을 기준으로 오름차순 정렬)
                    # 'Sort_Key'라는 임시 컬럼을 만들어 분(min) 단위 숫자로 변환
                    arr_raw['Sort_Key'] = arr_raw['STA'].apply(time_to_minutes)
                    dep_raw['Sort_Key'] = dep_raw['STD'].apply(time_to_minutes)

                    # 변환된 숫자를 기준으로 정렬 (오름차순)
                    arr_raw = arr_raw.sort_values(by='Sort_Key', ascending=True)
                    dep_raw = dep_raw.sort_values(by='Sort_Key', ascending=True)
                    
                    # 3. 보여줄 컬럼 선택
                    cols_arr = ['시간대', 'STA', 'ROUTE', 'ORGN', 'OPS', 'FLT NO']
                    cols_dep = ['시간대', 'STD', 'ROUTE', 'DEST', 'OPS', 'FLT NO']
                    
                    # 4. 스타일 적용
                    styled_arr = arr_raw[cols_arr].style.map(color_route_style, subset=['ROUTE'])
                    styled_dep = dep_raw[cols_dep].style.map(color_route_style, subset=['ROUTE'])
                    
                    # 5. 화면 표시 (2단 컬럼)
                    col_arr, col_dep = st.columns(2)
                    
                    with col_arr:
                        st.subheader("🛬 ICN 도착 (Arrival)")
                        st.dataframe(styled_arr, use_container_width=True, height=800, hide_index=True)
                        
                    with col_dep:
                        st.subheader("🛫 ICN 출발 (Departure)")
                        st.dataframe(styled_dep, use_container_width=True, height=800, hide_index=True)

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            import traceback
            st.text(traceback.format_exc())
    else:
        if 'analysis_done' in st.session_state:
            del st.session_state['analysis_done']
        st.info("파일을 업로드하고 분석을 시작하세요.")

# ==============================================================================
# 5. 메인 실행 로직: [두 스케줄 비교 분석 모드] (완벽 복구)
# ==============================================================================
elif analysis_mode == "두 스케줄 비교 분석":
    st.sidebar.header("⚙️ 비교 분석 설정")
    file1 = st.sidebar.file_uploader("📂 스케줄 1 (Before)", type="csv", key="file1")
    file2 = st.sidebar.file_uploader("📂 스케줄 2 (After)", type="csv", key="file2")
    
    if file1 and file2:
        try:
            df1 = load_data(file1)
            df2 = load_data(file2)
            
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
                    st.markdown("### 총 스케줄 가치 비교 (Scoring)")
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
                    st.dataframe(flt_cmp['time_changed'])
                    if not flt_cmp['removed'].empty: st.markdown("**삭제된 항공편**"); st.dataframe(flt_cmp['removed'])
                    if not flt_cmp['added'].empty: st.markdown("**신규 항공편**"); st.dataframe(flt_cmp['added'])
                
                with t3:
                    st.markdown("**사라진 연결**")
                    st.dataframe(conn_cmp['lost_connections'])
                    st.markdown("**새로운 연결**")
                    st.dataframe(conn_cmp['new_connections'])
                
                with t4:
                    st.markdown("**연결 시간/스코어 변경 상세**")
                    st.dataframe(conn_cmp['time_changes'])

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            import traceback
            st.text(traceback.format_exc())
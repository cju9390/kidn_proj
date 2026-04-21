import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Renal AKI CDSS",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.stApp { background-color: #f0f2f6; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background-color: #1e2235; border-right: none; }
section[data-testid="stSidebar"] > div { background: #1e2235 !important; }
section[data-testid="stSidebar"] * { color: #a0aec0; }
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important; border: none !important;
    color: #a0aec0 !important; text-align: center !important;
    border-radius: 8px !important; font-size: 0.82rem !important;
    padding: 10px 8px !important; box-shadow: none !important;
    width: 100% !important; margin-bottom: 2px !important;
    transition: background 0.15s, color 0.15s !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.08) !important; color: white !important;
}

/* ── Cards ── */
.card { background:white; border-radius:12px; padding:20px 24px; box-shadow:0 1px 4px rgba(0,0,0,0.08); margin-bottom:16px; }
.stat-card { background:white; border-radius:12px; padding:20px; box-shadow:0 1px 4px rgba(0,0,0,0.08); display:flex; justify-content:space-between; align-items:center; }
.stat-num { font-size:2rem; font-weight:700; color:#1a202c; line-height:1; }
.stat-label { font-size:0.8rem; color:#718096; margin-top:4px; }
.shortcut-card { background:white; border-radius:12px; padding:28px; box-shadow:0 1px 4px rgba(0,0,0,0.08); cursor:pointer; height:160px; position:relative; }
.shortcut-card:hover { box-shadow:0 4px 16px rgba(0,0,0,0.12); }
.shortcut-card.active { border:2px solid #48bb78; }

/* ── AKI 예측 ── */
.pred-card { border-radius:10px; padding:16px; text-align:center; border:1px solid #e2e8f0; background:white; }
.pred-horizon { font-size:0.75rem; color:#718096; font-weight:600; letter-spacing:1px; }
.pred-prob { font-size:2rem; font-weight:700; margin:6px 0 4px; }
.pred-badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; }

/* ── Alerts ── */
.alert-crit { background:#fff5f5; border-left:4px solid #fc8181; border-radius:6px; padding:12px 16px; margin-bottom:10px; }
.alert-warn { background:#fffaf0; border-left:4px solid #f6ad55; border-radius:6px; padding:12px 16px; margin-bottom:10px; }
.alert-info { background:#ebf8ff; border-left:4px solid #63b3ed; border-radius:6px; padding:12px 16px; margin-bottom:10px; }

/* ── Badges ── */
.badge-urgent { background:#fed7d7; color:#c53030; padding:2px 8px; border-radius:4px; font-size:0.72rem; font-weight:700; }
.badge-notice { background:#bee3f8; color:#2b6cb0; padding:2px 8px; border-radius:4px; font-size:0.72rem; font-weight:700; }
.badge-general { background:#c6f6d5; color:#276749; padding:2px 8px; border-radius:4px; font-size:0.72rem; font-weight:700; }

/* ── Nav ── */
.nav-btn { display:block; width:100%; padding:10px 14px; border-radius:8px; cursor:pointer; font-size:0.85rem; color:#4a5568; border:none; background:transparent; text-align:left; margin-bottom:2px; }
.nav-btn:hover { background:#edf2f7; }
.nav-btn.active { background:#ebf8ff; color:#2b6cb0; font-weight:600; }

/* ── Typography ── */
.page-title { font-size:1.6rem; font-weight:700; color:#1a202c; margin:0 0 4px 0; }
.page-sub { font-size:0.85rem; color:#718096; margin:0; }
.sec-title { font-size:0.95rem; font-weight:600; color:#2d3748; margin-bottom:12px; }

/* ── Steps ── */
.step-wrap { display:flex; align-items:center; gap:0; margin:16px 0; }
.step { display:flex; flex-direction:column; align-items:center; flex:1; }
.step-circle { width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.8rem; font-weight:700; }
.step-active { background:#3182ce; color:white; }
.step-done { background:#48bb78; color:white; }
.step-idle { background:#e2e8f0; color:#718096; }
.step-label { font-size:0.72rem; color:#718096; margin-top:4px; }
.step-line { flex:1; height:2px; background:#e2e8f0; margin-bottom:18px; }

/* ── Task rows ── */
.task-row { display:flex; align-items:center; gap:10px; padding:8px 0; border-bottom:1px solid #f7fafc; }
.task-dot-done { width:16px; height:16px; border-radius:50%; background:#e2e8f0; }
.task-dot-todo { width:16px; height:16px; border-radius:50%; border:2px solid #3182ce; background:transparent; }
.task-text-done { font-size:0.83rem; color:#a0aec0; text-decoration:line-through; }
.task-text-todo { font-size:0.83rem; color:#2d3748; }
.task-deadline { font-size:0.72rem; color:#3182ce; }

/* ── Guide rows ── */
.guide-row { display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #f7fafc; }
.guide-text { font-size:0.83rem; color:#2d3748; display:flex; align-items:center; gap:8px; }
.guide-btn { font-size:0.72rem; color:#3182ce; border:1px solid #90cdf4; background:#ebf8ff; border-radius:4px; padding:2px 10px; cursor:pointer; }
.ref-link { font-size:0.83rem; color:#3182ce; text-decoration:underline; cursor:pointer; display:block; margin-top:6px; }

/* ── Patient header ── */
.patient-headerbar { background:white; border-bottom:1px solid #e2e8f0; padding:10px 24px; font-size:0.83rem; display:flex; gap:24px; align-items:center; flex-wrap:wrap; }

/* ── Auth pages ── */
[data-testid="stForm"] {
    background: white !important; border: none !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.12) !important;
    padding: 40px !important;
}
.signup-form-row { display:flex; align-items:flex-start; padding:12px 0; border-bottom:1px solid #f0f2f6; gap:16px; }
.signup-label-col { min-width:110px; padding-top:10px; font-size:0.87rem; color:#4a5568; font-weight:500; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "login"
if "page" not in st.session_state:
    st.session_state.page = "홈"
if "selected_patient_idx" not in st.session_state:
    st.session_state.selected_patient_idx = 0


# ── Helpers ───────────────────────────────────────────────
def risk_color(v):
    if v >= 0.7: return "#e53e3e"
    if v >= 0.4: return "#dd6b20"
    return "#38a169"

def risk_label(v):
    if v >= 0.7: return "고위험", "#fed7d7", "#c53030"
    if v >= 0.4: return "중위험", "#feebc8", "#c05621"
    return "저위험", "#c6f6d5", "#276749"

def generate_vitals(seed=0, hours=24):
    rng = np.random.default_rng(seed)
    t = list(range(0, hours + 1, 2))
    sbp = np.clip(rng.normal(125, 8, len(t)), 90, 165)
    hr  = np.clip(rng.normal(80, 10, len(t)), 55, 130)
    temp = np.clip(rng.normal(36.8, 0.4, len(t)), 35.5, 39.5)
    return t, sbp.round(1), hr.round(0), temp.round(1)

def generate_cr_series(base_cr, seed=0):
    rng = np.random.default_rng(seed)
    t = list(range(-48, 1, 4))
    cr = [base_cr]
    for _ in t[1:]:
        cr.append(max(0.3, cr[-1] + rng.normal(0.05, 0.09)))
    return t, [round(v, 2) for v in cr]

def generate_future_cr(last_cr, seed=0):
    rng = np.random.default_rng(seed + 100)
    horizons = [6, 12, 24, 48]
    preds, v = [], last_cr
    for _ in horizons:
        v = max(0.3, v + rng.normal(0.12, 0.1))
        preds.append(round(v, 2))
    return horizons, preds

def generate_shap(seed=0):
    rng = np.random.default_rng(seed)
    features = ["현재 Creatinine", "ΔCr (48h)", "Urine Output", "BUN", "eGFR 변화율",
                 "나이", "수분 섭취", "MAP", "CRP", "Bicarbonate"]
    vals = rng.uniform(-0.35, 0.48, len(features))
    return features, vals


# ── Dummy Data ────────────────────────────────────────────
@st.cache_data
def generate_patients(n=20, seed=42):
    rng = np.random.default_rng(seed)
    ages = rng.integers(35, 82, n)
    sex = rng.choice(["M", "F"], n)
    base_cr = rng.uniform(0.7, 1.3, n)
    curr_cr = base_cr + rng.uniform(-0.05, 2.8, n)
    urine_output = rng.uniform(0.15, 2.5, n)
    bun = rng.uniform(12, 85, n)
    egfr = np.clip(186 * (curr_cr ** -1.154) * (ages ** -0.203), 5, 130)
    delta_cr = curr_cr - base_cr

    def kdigo(dc, uo):
        if dc >= 3.0 or uo < 0.3: return 3
        if dc >= 2.0 or uo < 0.5: return 2
        if dc >= 0.3 or uo < 1.0: return 1
        return 0

    stages = [kdigo(delta_cr[i], urine_output[i]) for i in range(n)]
    risk_base = np.clip(delta_cr / base_cr * 0.45 + (1 / (urine_output + 0.1)) * 0.35 + bun / 250 * 0.2, 0.05, 0.97)
    pred_6h  = np.clip(risk_base * rng.uniform(0.7,  1.0, n), 0.02, 0.95)
    pred_12h = np.clip(risk_base * rng.uniform(0.85, 1.1, n), 0.02, 0.97)
    pred_24h = np.clip(risk_base * rng.uniform(0.95, 1.2, n), 0.02, 0.98)
    pred_48h = np.clip(risk_base * rng.uniform(1.0,  1.3, n), 0.02, 0.99)

    wards = [f"중환자실-{rng.integers(1,5)}" if stages[i] >= 2
             else f"일반병동-{rng.integers(100,599)}" for i in range(n)]
    bdays = [f"{rng.integers(1955,1995)}-{rng.integers(1,12):02d}-{rng.integers(1,28):02d}" for _ in range(n)]
    reg_no = [f"A{rng.integers(100000,999999)}" for _ in range(n)]
    names_m = ["김민호", "이준서", "박상현", "최재원", "정민준", "강도현", "윤성우"]
    names_f = ["이지원", "박서연", "김민지", "정수아", "최예린", "강하은", "윤나은"]
    names = [rng.choice(names_m if sex[i] == "M" else names_f) for i in range(n)]

    return pd.DataFrame({
        "ID": [f"P{i+1:03d}" for i in range(n)],
        "이름": names, "나이": ages, "성별": sex, "생년월일": bdays,
        "등록번호": reg_no, "병동": wards,
        "기저 Cr": base_cr.round(2), "현재 Cr": curr_cr.round(2),
        "ΔCr": delta_cr.round(2), "Urine Output": urine_output.round(2),
        "BUN": bun.round(1), "eGFR": egfr.round(1), "KDIGO": stages,
        "Risk": risk_base.round(3),
        "AKI_6h": pred_6h.round(3), "AKI_12h": pred_12h.round(3),
        "AKI_24h": pred_24h.round(3), "AKI_48h": pred_48h.round(3),
    })

patients_df = generate_patients()

pages = ["홈", "마이", "대시보드", "원무", "일정"]

# ══════════════════════════════════════════════════════════
# 사이드바 (항상 렌더링 — auth/main 공통)
# ══════════════════════════════════════════════════════════
with st.sidebar:
    if st.session_state.logged_in:
        st.markdown("""
        <div style='padding:20px 0 12px; text-align:center;'>
            <div style='width:44px; height:44px; border-radius:50%; background:#3182ce;
                        display:flex; align-items:center; justify-content:center;
                        font-size:0.85rem; font-weight:700; color:white; margin:0 auto 8px;'>HG</div>
            <div style='font-size:0.7rem; color:#718096;'>홍길동</div>
        </div>
        <hr style='border-color:#2d3748; margin:0 0 12px 0;'>
        """, unsafe_allow_html=True)

        for name in pages:
            is_active = st.session_state.page == name
            if is_active:
                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.15); color:white; font-weight:600;"
                    f"border-radius:8px; padding:10px 8px; text-align:center; font-size:0.82rem;"
                    f"margin-bottom:2px;'>{name}</div>",
                    unsafe_allow_html=True)
            else:
                if st.button(name, key=f"nav_{name}", use_container_width=True):
                    st.session_state.page = name
                    st.rerun()

        st.markdown("<hr style='border-color:#2d3748; margin:12px 0;'>", unsafe_allow_html=True)
        if st.button("로그아웃", key="logout_btn", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.auth_page = "login"
            st.rerun()
    else:
        st.markdown("""
        <div style='padding:40px 16px; text-align:center;'>
            <div style='font-size:0.8rem; color:#718096; margin-top:8px; font-weight:600;'>Renal AKI CDSS</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# AUTH PAGES (로그인 / 회원가입)
# ══════════════════════════════════════════════════════════
if not st.session_state.logged_in:

    # ── 로그인 ─────────────────────────────────────────────
    if st.session_state.auth_page == "login":
        st.markdown("""<style>
        .stApp {
            background: linear-gradient(150deg, #aec8e0 0%, #c4d9ec 30%, #ddeaf5 60%, #c2d5e8 100%) !important;
        }
        [data-testid="stForm"] {
            background: white !important; border: none !important;
            border-radius: 16px !important;
            box-shadow: 0 8px 40px rgba(0,0,0,0.14) !important;
            padding: 44px 48px !important;
        }
        </style>""", unsafe_allow_html=True)

        # 상단 네비
        _, nav_r = st.columns([5, 1])
        with nav_r:
            n1, n2 = st.columns(2)
            with n1:
                st.markdown(
                    "<p style='text-align:center;padding-top:10px;font-weight:700;"
                    "font-size:0.9rem;color:#1a202c;'>로그인</p>",
                    unsafe_allow_html=True)
            with n2:
                if st.button("회원가입", key="top_to_signup", use_container_width=True):
                    st.session_state.auth_page = "signup"
                    st.rerun()

        # 중앙 카드
        _, mid, _ = st.columns([1, 1.4, 1])
        with mid:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            with st.form("login_form"):
                # 제목
                st.markdown("""
                <div style='text-align:center; margin-bottom:20px;'>
                    <div style='width:58px; height:58px; border-radius:50%; background:#1e2235;
                                display:inline-flex; align-items:center; justify-content:center;
                                font-size:1rem; font-weight:700; color:white; margin-bottom:14px;'>AKI</div>
                    <div style='font-size:1.3rem; font-weight:700; color:#1a202c;'>
                        로그인 &nbsp;▶
                    </div>
                </div>
                <hr style='border:none; border-top:1px solid #e2e8f0; margin:0 0 22px;'>
                """, unsafe_allow_html=True)

                sabun = st.text_input("사번", placeholder="사번", label_visibility="collapsed")
                pw    = st.text_input("비밀번호", placeholder="비밀번호", type="password", label_visibility="collapsed")

                lk_col, fg_col = st.columns([3, 3])
                with lk_col:
                    st.checkbox("로그인 상태 유지")
                with fg_col:
                    st.markdown(
                        "<div style='text-align:right;padding-top:6px;"
                        "font-size:0.82rem;color:#3182ce;cursor:pointer;'>"
                        "비밀번호를 잊으셨나요?</div>",
                        unsafe_allow_html=True)

                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                login_btn = st.form_submit_button("로그인", use_container_width=True, type="primary")

                st.markdown("""
                <hr style='border:none; border-top:1px solid #e2e8f0; margin:20px 0 10px;'>
                <p style='text-align:center; color:#718096; font-size:0.85rem; margin-bottom:2px;'>
                    새 사용자 회원가입?
                </p>
                """, unsafe_allow_html=True)

                if login_btn:
                    st.session_state.logged_in = True
                    st.session_state.page = "홈"
                    st.rerun()

            # 카드 아래 회원가입 버튼
            if st.button("회원가입", use_container_width=True, key="card_to_signup"):
                st.session_state.auth_page = "signup"
                st.rerun()

    # ── 회원가입 ────────────────────────────────────────────
    else:
        st.markdown("""<style>
        .stApp { background: #f0f2f6 !important; }
        [data-testid="stForm"] {
            background: white !important; border: none !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 6px rgba(0,0,0,0.08) !important;
            padding: 32px 40px !important;
        }
        </style>""", unsafe_allow_html=True)

        # 상단 네비
        _, nav_r = st.columns([5, 1])
        with nav_r:
            n1, n2 = st.columns(2)
            with n1:
                if st.button("로그인", key="top_to_login", use_container_width=True):
                    st.session_state.auth_page = "login"
                    st.rerun()
            with n2:
                st.markdown(
                    "<p style='text-align:center;padding-top:10px;font-weight:700;"
                    "font-size:0.9rem;color:#1a202c;'>회원가입</p>",
                    unsafe_allow_html=True)

        _, main_col, _ = st.columns([1, 4, 1])
        with main_col:
            st.markdown(
                "<h2 style='font-size:1.6rem;font-weight:700;color:#1a202c;margin-bottom:24px;'>"
                "신규 사원 회원가입</h2>",
                unsafe_allow_html=True)

            with st.form("signup_form"):
                st.markdown(
                    "<div style='font-size:1rem;font-weight:600;color:#2d3748;margin-bottom:20px;'>"
                    "기본 정보</div>",
                    unsafe_allow_html=True)

                def form_row(label, key, placeholder="", pw=False, hint=""):
                    lc, ic = st.columns([1, 4])
                    with lc:
                        st.markdown(
                            f"<div style='padding-top:10px;font-size:0.87rem;"
                            f"color:#4a5568;font-weight:500;'>{label}</div>",
                            unsafe_allow_html=True)
                    with ic:
                        if pw:
                            val = st.text_input(label, key=key, placeholder=placeholder,
                                                type="password", label_visibility="collapsed")
                        else:
                            val = st.text_input(label, key=key, placeholder=placeholder,
                                                label_visibility="collapsed")
                        if hint:
                            st.markdown(
                                f"<div style='font-size:0.75rem;color:#a0aec0;margin-top:2px;'>{hint}</div>",
                                unsafe_allow_html=True)
                    return val

                form_row("성명", "su_name", "성명")
                form_row("사번", "su_id", "사번")
                form_row("비밀번호", "su_pw", "비밀번호를 입력해주세요", pw=True,
                         hint="최소 8자 이상 12글자 이내, 대소문자 혼합 특수문자 포함")
                form_row("비밀번호 확인", "su_pw2", "비밀번호를 다시 입력해주세요", pw=True)
                form_row("주민등록번호", "su_rrn", "주민등록번호")
                form_row("전화번호", "su_tel", "")
                form_row("이메일", "su_email", "이메일")

                # 소속 부서 / 직급
                lc1, ic1 = st.columns([1, 4])
                with lc1:
                    st.markdown(
                        "<div style='padding-top:10px;font-size:0.87rem;color:#4a5568;font-weight:500;'>"
                        "소속 부서</div>", unsafe_allow_html=True)
                with ic1:
                    st.selectbox("소속 부서", ["원무과", "간호부", "신장내과", "중환자실", "응급의학과", "기타"],
                                 key="su_dept", label_visibility="collapsed")

                lc2, ic2 = st.columns([1, 4])
                with lc2:
                    st.markdown(
                        "<div style='padding-top:10px;font-size:0.87rem;color:#4a5568;font-weight:500;'>"
                        "직급</div>", unsafe_allow_html=True)
                with ic2:
                    st.selectbox("직급", ["직급 선택", "교수", "전공의", "인턴", "간호사", "원무직원", "기타"],
                                 key="su_rank", label_visibility="collapsed")

                st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

                bc1, bc2, bc3 = st.columns([2, 1, 3])
                with bc1:
                    submitted = st.form_submit_button("회원가입 완료", use_container_width=True, type="primary")
                with bc2:
                    cancelled = st.form_submit_button("취소", use_container_width=True)

                if submitted:
                    st.success("회원가입이 완료되었습니다! 관리자 승인 후 로그인이 가능합니다.")
                if cancelled:
                    st.session_state.auth_page = "login"
                    st.rerun()

    st.stop()


# ── 페이지 라우팅 ──────────────────────────────────────────
page = st.session_state.page


# ══════════════════════════════════════════════════════════
# 홈
# ══════════════════════════════════════════════════════════
if page == "홈":
    col_title, col_top = st.columns([3, 1])
    with col_title:
        st.markdown("<p class='page-title'>환영합니다, 홍길동님</p>", unsafe_allow_html=True)
        st.markdown("<p class='page-sub'>오늘도 안전한 환자 관리를 위해 함께합니다. (현재 접속: 중환자실 / 원무과)</p>", unsafe_allow_html=True)
    with col_top:
        st.markdown(
            f"<div style='text-align:right; padding-top:8px;'>"
            f"<span style='font-size:0.85rem; color:#718096;'>{datetime.now().strftime('%Y년 %m월 %d일')}</span>"
            f"</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    stats = [
        ("오늘의 입원 환자", "142", "+3", "#3182ce"),
        ("중환자실 재실", "28", "FULL", "#e53e3e"),
        ("신규 원무 등록", "45", "", "#38a169"),
        ("CDSS 활성 알림", "12", "주의", "#dd6b20"),
    ]
    for col, (label, num, badge, color) in zip([sc1, sc2, sc3, sc4], stats):
        with col:
            if badge == "FULL":
                badge_html = "<span style='background:#fed7d7;color:#c53030;font-size:0.65rem;font-weight:700;padding:1px 6px;border-radius:4px;margin-left:6px;'>FULL</span>"
            elif badge == "주의":
                badge_html = "<span style='background:#feebc8;color:#c05621;font-size:0.65rem;font-weight:700;padding:1px 6px;border-radius:4px;margin-left:6px;'>주의</span>"
            elif badge:
                badge_html = f"<span style='color:#38a169;font-size:0.75rem;margin-left:4px;'>{badge}</span>"
            else:
                badge_html = ""
            st.markdown(f"""
            <div class='stat-card'>
                <div>
                    <div class='stat-label'>{label}</div>
                    <div class='stat-num' style='color:{color};'>{num}{badge_html}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='sec-title'>주요 시스템 바로가기</p>", unsafe_allow_html=True)

    lk1, lk2 = st.columns(2)
    with lk1:
        st.markdown("""
        <div class='shortcut-card'>
            <div style='font-size:1rem;font-weight:600;color:#1a202c;margin-bottom:8px;'>CDSS 환자 대시보드</div>
            <div style='font-size:0.82rem;color:#718096;line-height:1.5;'>
                AI 기반 신장 이식 환자 중환자실 급성 신손상(AKI) 조기 예측 및 임상 의사 결정 지원 시스템으로 이동합니다.
            </div>
            <div style='position:absolute;top:20px;right:20px;color:#a0aec0;'>›</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("대시보드 이동", key="home_to_dash", use_container_width=True):
            st.session_state.page = "대시보드"
            st.rerun()

    with lk2:
        st.markdown("""
        <div class='shortcut-card active'>
            <div style='font-size:1rem;font-weight:600;color:#1a202c;margin-bottom:8px;'>환자 원무 등록</div>
            <div style='font-size:0.82rem;color:#718096;line-height:1.5;'>
                신규 환자 정보 등록, 진료과 배정 및 입원 수속을 위한 원무 관리 시스템으로 이동합니다.
            </div>
            <div style='position:absolute;top:20px;right:20px;color:#a0aec0;'>›</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("원무 이동", key="home_to_admin", use_container_width=True):
            st.session_state.page = "원무"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    bot_l, bot_r = st.columns([3, 2])

    with bot_l:
        st.markdown("""
        <div class='card'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;'>
                <span class='sec-title' style='margin-bottom:0;'>최근 공지사항</span>
                <span style='font-size:0.8rem;color:#3182ce;cursor:pointer;'>전체보기</span>
            </div>
            <div style='display:flex;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid #f7fafc;'>
                <span class='badge-urgent'>긴급</span>
                <div>
                    <div style='font-size:0.85rem;color:#1a202c;'>CDSS 시스템 서버 점검 안내 (4/20 02:00~04:00)</div>
                    <div style='font-size:0.75rem;color:#a0aec0;'>전산정보팀 · 2시간 전</div>
                </div>
            </div>
            <div style='display:flex;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid #f7fafc;'>
                <span class='badge-notice'>안내</span>
                <div>
                    <div style='font-size:0.85rem;color:#1a202c;'>신규 입원 환자 병상 배정 프로세스 변경 건</div>
                    <div style='font-size:0.75rem;color:#a0aec0;'>원무팀 · 어제</div>
                </div>
            </div>
            <div style='display:flex;align-items:center;gap:10px;padding:10px 0;'>
                <span class='badge-general'>일반</span>
                <div>
                    <div style='font-size:0.85rem;color:#1a202c;'>중환자실 면회 시간 임시 조정 안내</div>
                    <div style='font-size:0.75rem;color:#a0aec0;'>간호부 · 3일 전</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with bot_r:
        st.markdown("""
        <div class='card'>
            <div class='sec-title'>나의 업무 현황</div>
            <div class='task-row'>
                <div class='task-dot-done'></div>
                <div>
                    <div class='task-text-done'>오전 병동 라운딩</div>
                    <div style='font-size:0.72rem;color:#a0aec0;'>완료됨</div>
                </div>
            </div>
            <div class='task-row'>
                <div class='task-dot-todo'></div>
                <div>
                    <div class='task-text-todo'>김민호 환자 CDSS 경고 확인</div>
                    <div class='task-deadline'>14:00 까지 마감</div>
                </div>
            </div>
            <div class='task-row'>
                <div class='task-dot-todo'></div>
                <div>
                    <div class='task-text-todo'>주간 원무 통계 리포트 작성</div>
                    <div class='task-deadline'>18:00 까지 마감</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# 마이페이지
# ══════════════════════════════════════════════════════════
elif page == "마이":
    st.markdown("<p class='page-title'>마이페이지 (내 정보)</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col_form, _ = st.columns([2, 3])
    with col_form:
        st.markdown("""
        <div class='card'>
            <div class='sec-title'>계정 정보</div>
            <table style='width:100%;border-collapse:collapse;'>
                <tr style='border-bottom:1px solid #f0f0f0;'>
                    <td style='padding:16px 0;color:#718096;font-size:0.87rem;width:120px;'>이름</td>
                    <td style='padding:16px 0;color:#1a202c;font-size:0.87rem;'>홍길동</td>
                </tr>
                <tr style='border-bottom:1px solid #f0f0f0;'>
                    <td style='padding:16px 0;color:#718096;font-size:0.87rem;'>사번</td>
                    <td style='padding:16px 0;color:#3182ce;font-size:0.87rem;font-weight:600;'>EMP-20260419</td>
                </tr>
                <tr style='border-bottom:1px solid #f0f0f0;'>
                    <td style='padding:16px 0;color:#718096;font-size:0.87rem;'>비밀번호</td>
                    <td style='padding:16px 0;'>
                        <span style='letter-spacing:3px;color:#1a202c;'>········</span>
                        <button style='margin-left:16px;background:white;border:1px solid #e2e8f0;
                                       border-radius:6px;padding:3px 12px;font-size:0.78rem;cursor:pointer;color:#4a5568;'>수정 / 확인</button>
                    </td>
                </tr>
                <tr>
                    <td style='padding:16px 0;color:#718096;font-size:0.87rem;'>소속 부서</td>
                    <td style='padding:16px 0;color:#1a202c;font-size:0.87rem;'>원무과</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CDSS 대시보드
# ══════════════════════════════════════════════════════════
elif page == "대시보드":
    with st.sidebar:
        st.markdown("<hr style='border-color:#2d3748;margin:8px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.75rem;color:#718096;padding:4px 0;'>환자 선택</div>", unsafe_allow_html=True)
        pat_options = [f"{r['이름']} ({r['등록번호']})" for _, r in patients_df.iterrows()]
        sel_str = st.selectbox("환자", pat_options, label_visibility="collapsed",
                               index=st.session_state.selected_patient_idx)
        st.session_state.selected_patient_idx = pat_options.index(sel_str)

    sel_idx = st.session_state.selected_patient_idx
    p = patients_df.iloc[sel_idx]

    st.markdown(f"""
    <div style='background:white;border-bottom:1px solid #e2e8f0;padding:10px 0 10px;
                display:flex;gap:28px;align-items:center;flex-wrap:wrap;margin-bottom:16px;'>
        <span style='font-size:0.95rem;font-weight:700;color:#1a202c;margin-right:4px;'>CDSS</span>
        <span><span style='color:#718096;font-size:0.78rem;'>이름 </span>
              <strong style='font-size:0.85rem;'>{p['이름']}</strong></span>
        <span><span style='color:#718096;font-size:0.78rem;'>연령 </span>
              <strong style='font-size:0.85rem;'>{p['나이']}세 {"남성" if p["성별"]=="M" else "여성"}</strong></span>
        <span><span style='color:#718096;font-size:0.78rem;'>등록번호 </span>
              <strong style='font-size:0.85rem;'>{p['등록번호']}</strong></span>
        <span><span style='color:#718096;font-size:0.78rem;'>생년월일 </span>
              <strong style='font-size:0.85rem;'>{p['생년월일']}</strong></span>
        <span><span style='color:#718096;font-size:0.78rem;'>병동 </span>
              <strong style='font-size:0.85rem;'>{p['병동']}</strong></span>
        <span style='background:#fed7d7;color:#c53030;padding:3px 10px;border-radius:6px;
                     font-size:0.78rem;font-weight:600;margin-left:auto;'>알레르기: 페니실린</span>
        <span style='background:#c6f6d5;color:#276749;padding:3px 10px;border-radius:6px;
                     font-size:0.78rem;font-weight:600;'>활력 징후: 안정</span>
    </div>
    """, unsafe_allow_html=True)

    sub_nav, col_main, col_right = st.columns([1.3, 3.5, 2.2])

    with sub_nav:
        st.markdown("""
        <div class='card' style='padding:12px 8px;'>
            <div class='nav-btn active'>홈</div>
            <div class='nav-btn'>종합 정보</div>
            <div class='nav-btn'>환자 검색</div>
            <div class='nav-btn'>문제 목록</div>
            <div class='nav-btn'>처방 약물</div>
            <div class='nav-btn'>검사 결과</div>
            <div class='nav-btn'>영상 의학</div>
            <div class='nav-btn'>간호 계획</div>
            <div class='nav-btn'>카디오 필터</div>
        </div>
        """, unsafe_allow_html=True)

    with col_main:
        # 활력 징후
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            top_l, top_r = st.columns([3, 1])
            with top_l:
                st.markdown("<div class='sec-title'>활력 징후 추이</div>", unsafe_allow_html=True)
            with top_r:
                vh = st.selectbox("기간", ["지난 24시간", "지난 12시간", "지난 6시간"],
                                  label_visibility="collapsed", key="vital_range")
            hours_n = 24 if "24" in vh else 12 if "12" in vh else 6
            vt, sbp, hr_v, temp = generate_vitals(seed=sel_idx, hours=hours_n)
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(x=vt, y=sbp, name="혈압(SBP)",
                                       line=dict(color="#3182ce", width=2), mode="lines+markers", marker=dict(size=4)))
            fig_v.add_trace(go.Scatter(x=vt, y=hr_v, name="심박수",
                                       line=dict(color="#e53e3e", width=2), mode="lines+markers", marker=dict(size=4)))
            fig_v.add_trace(go.Scatter(x=vt, y=temp * 3, name="체온(×3)",
                                       line=dict(color="#dd6b20", width=2, dash="dot"), mode="lines", opacity=0.7))
            fig_v.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
                                 margin=dict(t=4, b=4, l=4, r=4),
                                 legend=dict(orientation="h", y=1.15, font=dict(size=11)),
                                 xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                                 yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                                 font=dict(size=11))
            st.plotly_chart(fig_v, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # AKI 예측
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>AKI 발생 예측 (다중 시간대)</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.78rem;color:#718096;margin-bottom:12px;'>XGBoost · MIMIC-IV 기반 · KDIGO 2012 기준 | 더미 예측값</div>", unsafe_allow_html=True)
        horizons = ["6시간", "12시간", "24시간", "48시간"]
        probs    = [float(p["AKI_6h"]), float(p["AKI_12h"]), float(p["AKI_24h"]), float(p["AKI_48h"])]
        colors_h = [risk_color(v) for v in probs]
        pc1, pc2, pc3, pc4 = st.columns(4)
        for col, horizon, prob in zip([pc1, pc2, pc3, pc4], horizons, probs):
            lbl, bg, fg = risk_label(prob)
            with col:
                st.markdown(f"""
                <div class='pred-card'>
                    <div class='pred-horizon'>{horizon} 예측</div>
                    <div class='pred-prob' style='color:{fg};'>{prob*100:.1f}%</div>
                    <span class='pred-badge' style='background:{bg};color:{fg};'>{lbl}</span>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Bar(x=horizons, y=[v * 100 for v in probs], marker_color=colors_h,
                                  text=[f"{v*100:.1f}%" for v in probs], textposition="outside",
                                  hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
        fig_pred.add_hline(y=70, line_dash="dash", line_color="#e53e3e",
                           annotation_text="고위험 임계값 (70%)", annotation_font_color="#e53e3e",
                           annotation_position="bottom right")
        fig_pred.add_hline(y=40, line_dash="dash", line_color="#dd6b20",
                           annotation_text="중위험 임계값 (40%)", annotation_font_color="#dd6b20",
                           annotation_position="bottom right")
        fig_pred.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
                                margin=dict(t=24, b=8, l=4, r=4),
                                yaxis=dict(range=[0, 105], title="%", showgrid=True, gridcolor="#f0f0f0"),
                                xaxis=dict(showgrid=False), font=dict(size=11), showlegend=False)
        st.plotly_chart(fig_pred, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 위험도 게이지
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>위험도 점수</div>", unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)

        def gauge(val, title):
            color = risk_color(val)
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=round(val * 100, 1),
                title={"text": title, "font": {"size": 12, "color": "#4a5568"}},
                number={"suffix": "%", "font": {"size": 22, "color": color}},
                gauge={"axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#e2e8f0",
                                "tickfont": {"size": 9, "color": "#a0aec0"}},
                       "bar": {"color": color, "thickness": 0.25}, "bgcolor": "#f7fafc", "borderwidth": 0,
                       "steps": [{"range": [0, 40], "color": "rgba(56,161,105,0.1)"},
                                  {"range": [40, 70], "color": "rgba(221,107,32,0.1)"},
                                  {"range": [70, 100], "color": "rgba(229,62,62,0.1)"}]}))
            fig.update_layout(height=160, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=24, b=0, l=12, r=12))
            return fig

        with g1:
            st.plotly_chart(gauge(float(p["Risk"]), "종합 위험"), use_container_width=True)
            kdigo = int(p["KDIGO"])
            kdigo_colors = {0: ("#c6f6d5","#276749"), 1: ("#feebc8","#c05621"),
                            2: ("#fed7d7","#c53030"), 3: ("#e9d8fd","#6b46c1")}
            bg_k, fg_k = kdigo_colors[kdigo]
            st.markdown(f"<div style='text-align:center;margin-top:-8px;'>"
                        f"<span style='background:{bg_k};color:{fg_k};font-size:0.78rem;"
                        f"font-weight:700;padding:2px 12px;border-radius:20px;'>"
                        f"KDIGO Stage {kdigo}</span></div>", unsafe_allow_html=True)
        with g2:
            st.plotly_chart(gauge(0.40, "낙상 위험"), use_container_width=True)
        with g3:
            st.plotly_chart(gauge(0.20, "욕창 위험"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Creatinine 추이
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>Creatinine 추이 및 예측</div>", unsafe_allow_html=True)
        cr_t, cr_v = generate_cr_series(float(p["기저 Cr"]), seed=sel_idx)
        fut_t, fut_v = generate_future_cr(cr_v[-1], seed=sel_idx)
        fig_cr = go.Figure()
        fig_cr.add_hline(y=float(p["기저 Cr"]) * 1.5, line_dash="dash", line_color="#dd6b20", line_width=1,
                         annotation_text="KDIGO 1 (×1.5)", annotation_font_color="#dd6b20", annotation_position="right")
        fig_cr.add_hline(y=float(p["기저 Cr"]) * 2.0, line_dash="dash", line_color="#e53e3e", line_width=1,
                         annotation_text="KDIGO 2 (×2.0)", annotation_font_color="#e53e3e", annotation_position="right")
        fig_cr.add_trace(go.Scatter(x=cr_t, y=cr_v, name="실측",
                                    line=dict(color="#3182ce", width=2), mode="lines+markers", marker=dict(size=5)))
        fig_cr.add_trace(go.Scatter(x=[cr_t[-1]] + fut_t, y=[cr_v[-1]] + fut_v, name="AI 예측",
                                    line=dict(color="#e53e3e", width=2, dash="dot"),
                                    mode="lines+markers", marker=dict(size=6, symbol="diamond", color="#e53e3e")))
        fig_cr.add_vline(x=0, line_dash="dash", line_color="#a0aec0", line_width=1,
                         annotation_text="현재", annotation_font_color="#a0aec0")
        fig_cr.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
                              margin=dict(t=8, b=8, l=4, r=80),
                              xaxis=dict(title="시간 (h, 0=현재)", showgrid=True, gridcolor="#f0f0f0"),
                              yaxis=dict(title="Cr (mg/dL)", showgrid=True, gridcolor="#f0f0f0"),
                              legend=dict(orientation="h", y=1.15, font=dict(size=11)), font=dict(size=11))
        st.plotly_chart(fig_cr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # SHAP
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>SHAP 피처 기여도</div>", unsafe_allow_html=True)
        feats, shap_v = generate_shap(seed=sel_idx)
        sidx = np.argsort(np.abs(shap_v))[::-1]
        fs = [feats[i] for i in sidx]
        sv = [shap_v[i] for i in sidx]
        fig_shap = go.Figure(go.Bar(x=sv, y=fs, orientation="h",
                                    marker_color=["#e53e3e" if v > 0 else "#3182ce" for v in sv],
                                    hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig_shap.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
                                margin=dict(t=4, b=4, l=4, r=4),
                                xaxis=dict(title="SHAP value", showgrid=True, gridcolor="#f0f0f0",
                                           zeroline=True, zerolinecolor="#e2e8f0"),
                                yaxis=dict(showgrid=False), font=dict(size=11))
        st.plotly_chart(fig_shap, use_container_width=True)
        st.markdown("<div style='font-size:0.75rem;color:#a0aec0;'>빨강: 위험 상승 기여 &nbsp;|&nbsp; 파랑: 위험 감소 기여 &nbsp;|&nbsp; ※ 더미 SHAP값</div>",
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>권고 및 알림</div>", unsafe_allow_html=True)
        risk_v = float(p["Risk"])
        if risk_v >= 0.7:
            st.markdown(f"""
            <div class='alert-crit'>
                <div style='font-size:0.8rem;font-weight:600;color:#c53030;'>약물 상호작용 경고</div>
                <div style='font-size:0.77rem;color:#742a2a;margin-top:4px;'>
                    처방된 와파린과 이부프로펜 간의 잠재적 상호작용. 출혈 위험 증가.
                </div>
            </div>
            <div class='alert-warn'>
                <div style='font-size:0.8rem;font-weight:600;color:#c05621;'>중복 처방 알림</div>
                <div style='font-size:0.77rem;color:#7b341e;margin-top:4px;'>
                    최근 48시간 이내 복부/골반 CT 스캔이 이미 처방됨.
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif risk_v >= 0.4:
            st.markdown(f"""
            <div class='alert-warn'>
                <div style='font-size:0.8rem;font-weight:600;color:#c05621;'>AKI 위험 중등도</div>
                <div style='font-size:0.77rem;color:#7b341e;margin-top:4px;'>
                    AKI 24h 예측 {float(p["AKI_24h"])*100:.1f}%. 4시간 내 재평가 권장.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='alert-info'>
                <div style='font-size:0.8rem;font-weight:600;color:#2b6cb0;'>현재 안정적</div>
                <div style='font-size:0.77rem;color:#2c5282;margin-top:4px;'>
                    위험 지표 정상 범위. 정기 모니터링 유지.
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>임상 가이드라인</div>", unsafe_allow_html=True)
        for g in ["항생제 요법 시작 (피페라실린-타조박탐)", "젖산 수치 확인", "정맥 수액 투여"]:
            st.markdown(f"""
            <div class='guide-row'>
                <div class='guide-text'>■ {g}</div>
                <button class='guide-btn'>처방</button>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>AI 분석 및 참고 자료</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size:0.82rem;color:#4a5568;line-height:1.6;background:#f7fafc;
                    border-radius:8px;padding:12px;margin-bottom:12px;'>
            최근 백혈구 수치 증가와 발열 스파이크는 초기 패혈증을 강력히 시사합니다.<br><br>
            AKI 48h 예측: <strong style='color:{risk_color(float(p["AKI_48h"]))}'>{float(p["AKI_48h"])*100:.1f}%</strong> —
            {'즉각적인 신장 보호 중재 권장.' if float(p["AKI_48h"]) >= 0.7 else '4~6시간 후 재평가 권장.'}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <a class='ref-link'>SCCM 패혈증 가이드라인 2021</a>
        <a class='ref-link'>패혈증 바이오마커에 대한 최신 연구</a>
        <a class='ref-link'>KDIGO AKI 가이드라인 2012</a>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 고위험 환자 테이블
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>고위험 환자 목록 (전체)</div>", unsafe_allow_html=True)
    disp = patients_df[["ID", "이름", "나이", "성별", "병동", "현재 Cr", "ΔCr",
                         "Urine Output", "eGFR", "KDIGO", "Risk",
                         "AKI_6h", "AKI_12h", "AKI_24h", "AKI_48h"]].copy()
    disp = disp.sort_values("Risk", ascending=False).reset_index(drop=True)

    def color_risk_cell(v):
        if v >= 0.7: return "color: #c53030; font-weight: 600"
        if v >= 0.4: return "color: #c05621"
        return "color: #276749"

    def color_kdigo_cell(v):
        if v >= 3: return "color: #6b46c1; font-weight: 600"
        if v >= 2: return "color: #c53030"
        if v >= 1: return "color: #c05621"
        return "color: #276749"

    styled = (disp.style
              .applymap(color_risk_cell, subset=["Risk", "AKI_6h", "AKI_12h", "AKI_24h", "AKI_48h"])
              .applymap(color_kdigo_cell, subset=["KDIGO"])
              .format({"Risk": "{:.1%}", "AKI_6h": "{:.1%}", "AKI_12h": "{:.1%}",
                       "AKI_24h": "{:.1%}", "AKI_48h": "{:.1%}",
                       "ΔCr": "{:+.2f}", "Urine Output": "{:.2f}", "eGFR": "{:.0f}"}))
    st.dataframe(styled, use_container_width=True, hide_index=True, height=300)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# 원무
# ══════════════════════════════════════════════════════════
elif page == "원무":
    st.markdown("""
    <div style='background:white;border-bottom:1px solid #e2e8f0;padding:10px 0;
                font-size:0.85rem;font-weight:600;color:#1a202c;margin-bottom:16px;'>
        환자 원무 관리 시스템
    </div>
    """, unsafe_allow_html=True)

    sub_col, main_col = st.columns([1.2, 4])
    with sub_col:
        st.markdown("""
        <div class='card' style='padding:12px 8px;'>
            <div style='display:flex;align-items:center;gap:8px;padding:8px;margin-bottom:8px;'>
                <div style='font-size:0.8rem;font-weight:600;color:#1a202c;'>환자 원무 관리 시스템</div>
            </div>
            <div class='nav-btn active'>신규 환자 등록</div>
            <div class='nav-btn'>내원 관리</div>
            <div class='nav-btn'>예약 현황</div>
            <div class='nav-btn'>서류 발급</div>
            <div class='nav-btn'>통계 및 보고서</div>
        </div>
        """, unsafe_allow_html=True)

    with main_col:
        top3 = st.columns(3)
        with top3[0]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-title'>환자 기본 정보</div>", unsafe_allow_html=True)
            st.text_input("이름", placeholder="이름", label_visibility="collapsed", key="reg_name")
            st.text_input("주민등록번호", placeholder="주민등록번호", label_visibility="collapsed", key="reg_rrn")
            st.text_input("전화번호", placeholder="전화번호", label_visibility="collapsed", key="reg_tel")
            st.text_input("주소", placeholder="주소", label_visibility="collapsed", key="reg_addr")
            st.markdown("</div>", unsafe_allow_html=True)
        with top3[1]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-title'>보호자 및 긴급 연락처</div>", unsafe_allow_html=True)
            st.text_input("보호자 성함", placeholder="보호자 성함", label_visibility="collapsed", key="reg_guardian")
            st.text_input("보호자 전화번호", placeholder="보호자 전화번호", label_visibility="collapsed", key="reg_gtel")
            st.selectbox("관계", ["관계", "배우자", "부모", "자녀", "형제", "기타"],
                         label_visibility="collapsed", key="reg_rel")
            st.text_input("긴급 연락처 (선택)", placeholder="긴급 연락처 (선택)", label_visibility="collapsed", key="reg_emg")
            st.markdown("</div>", unsafe_allow_html=True)
        with top3[2]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-title'>내원 사유 및 보험 정보</div>", unsafe_allow_html=True)
            st.text_area("내원 사유", height=90, label_visibility="collapsed", key="reg_reason")
            st.selectbox("보험 구분", ["건강보험", "의료급여1종", "의료급여2종", "자동차보험", "산재", "비급여"],
                         label_visibility="collapsed", key="reg_ins")
            st.markdown("</div>", unsafe_allow_html=True)

        bot3_l, bot3_r = st.columns([3, 2])
        with bot3_l:
            top_b = st.columns(2)
            with top_b[0]:
                st.markdown("""
                <div class='card'>
                    <div class='sec-title'>신규 환자 등록 진행 현황</div>
                    <div class='step-wrap'>
                        <div class='step'><div class='step-circle step-active'>1</div><div class='step-label'>정보 입력</div></div>
                        <div class='step-line'></div>
                        <div class='step'><div class='step-circle step-idle'>2</div><div class='step-label'>서류 제출</div></div>
                        <div class='step-line'></div>
                        <div class='step'><div class='step-circle step-idle'>3</div><div class='step-label'>최종 확인</div></div>
                        <div class='step-line'></div>
                        <div class='step'><div class='step-circle step-idle'>4</div><div class='step-label'>등록 완료</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                c_btn1, c_btn2 = st.columns(2)
                with c_btn1:
                    st.button("등록 완료", type="primary", use_container_width=True)
                with c_btn2:
                    st.button("취소", use_container_width=True)
            with top_b[1]:
                st.markdown("""
                <div class='card'>
                    <div class='sec-title'>등록 이력</div>
                    <div style='font-size:0.8rem;color:#4a5568;line-height:1.8;max-height:160px;overflow-y:auto;'>
                        <p>* 최근 방문: 2026.04.10 정형외과 외래</p>
                        <p>* 행정: 개인정보 수집 동의서 갱신 필요</p>
                        <p>* 이전 입원: 2025.11.03 ~ 2025.11.17 (신장내과)</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with bot3_r:
            st.markdown("""<div class='card'><div class='sec-title'>빠른 등록 기능</div>""", unsafe_allow_html=True)
            for btn_label in ["주민등록번호 자동 스캔", "과거 환자 정보 불러오기", "보호자 정보 복사"]:
                st.button(btn_label, use_container_width=True, key=f"quick_{btn_label}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class='card'>
                <div class='sec-title'>업무 알림 및 할 일</div>
                <div style='display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #f7fafc;'>
                    <span style='font-size:0.82rem;color:#2d3748;'>서류 미제출 건</span>
                    <span style='background:#fed7d7;color:#c53030;padding:1px 8px;border-radius:10px;font-size:0.75rem;font-weight:700;'>2</span>
                </div>
                <div style='padding:8px 0;border-bottom:1px solid #f7fafc;'>
                    <span style='font-size:0.82rem;color:#2d3748;'>재방문 환자 확인</span>
                </div>
                <div style='padding:8px 0;'>
                    <span style='font-size:0.82rem;color:#2d3748;'>VIP 환자 응대 매뉴얼</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# 일정
# ══════════════════════════════════════════════════════════
elif page == "일정":
    st.markdown("<p class='page-title'>일정</p>", unsafe_allow_html=True)
    st.markdown("<div class='card'><p style='color:#718096;text-align:center;padding:40px;'>일정 기능은 준비 중입니다.</p></div>",
                unsafe_allow_html=True)

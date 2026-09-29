"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
xgb_model/feature_config.py  —  피처 목록 & 전처리 설정 중앙 관리
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ─────────────────────────────────────────────────────────────────────────────
# 타겟 및 제외 컬럼
# ─────────────────────────────────────────────────────────────────────────────

TARGET = "aki_label"

EXCLUDE_COLS = [
    "stay_id", "subject_id", "hadm_id",
    "aki_label", "aki_stage", "aki_onset_time",
    "prediction_cutoff", "effective_cutoff",
    "hours_to_aki",
    "is_pseudo_cutoff",
    "icu_intime", "icu_outtime",
    "hospital_expire_flag",
    "competed_with_death",
    "hours_to_death",
    "death_time",
    "icu_los_hours",       # 미래 정보 (단독 AUROC 0.7238)
    "map_missing",         # 윈도우 길이 누수 (단독 AUROC 0.9019)
    "hemoglobin_missing",  # 윈도우 길이 누수 (단독 AUROC 0.9050)
    "shock_index_missing", # 윈도우 길이 누수 (단독 AUROC 0.8719)
]


# ─────────────────────────────────────────────────────────────────────────────
# Track A  —  혈액검사
# ─────────────────────────────────────────────────────────────────────────────
FEAT_LAB = [
    "cr_min", "cr_max", "cr_mean", "cr_delta",
    "bun_max", "bun_mean", "bun_cr_ratio",
    "potassium_max", "potassium_mean",
    "bicarbonate_min", "bicarbonate_mean",
    "hemoglobin_min", "hemoglobin_mean",
    "lactate_max", "lactate_mean",
    "egfr_ckdepi",
]

# Track A  —  활력징후
# ★ 수정: vital_map_min·vital_map_mean·vital_shock_index 제거
#   이유: FEAT_ISCHEMIC의 isch_map_min·isch_map_mean·isch_shock_index와
#         동일 소스·동일 계산 → 중복 피처
#         step7_vital_urine_features.sql에서 HR·RR·SBP·Temp·SpO2만 신규 추가
FEAT_VITAL = [
    "hr_min", "hr_max", "hr_mean",   # 심박수
    "rr_max", "rr_mean",             # 호흡수 (rr_max>30 = 과호흡 = 대사산증 신호)
    "sbp_min", "sbp_mean",           # 수축기혈압 (sbp_min<90 = 저혈압 = AKI 위험)
    "temp_min", "temp_mean",         # 체온°C (step7 SQL에서 F→C 자동 변환)
    "spo2_min", "spo2_mean",         # 산소포화도 (spo2_min<90 = 저산소)
    # vital_map_min, vital_map_mean  → isch_map_min, isch_map_mean (FEAT_ISCHEMIC)
    # vital_shock_index              → isch_shock_index             (FEAT_ISCHEMIC)
]

# Track A  —  소변량·수액 균형 (step7_vital_urine_features.sql 실행 후 활성화)
FEAT_URINE = [
    "urine_6h",            # 입실 후 6시간 소변량 (mL) — KDIGO 조기 핍뇨 기준
    "urine_24h",           # 24시간 누적 소변량 (mL)
    "urine_total",         # effective_cutoff까지 전체 소변량 (mL)
    "urine_zero_ratio",    # 소변 없는 시간 비율 (0~1, 1=무뇨)
    "fluid_24h",           # 24시간 수액 투여량 (mL)
    "fluid_total",         # 전체 수액 투여량 (mL)
    "fluid_balance_24h",   # 24h 수액 균형 = 투여 - 소변 (양수=과수액)
    "fluid_balance_total", # 전체 수액 균형 (mL)
]

# Track A  —  결측 지시변수
# map_missing 제거 → EXCLUDE_COLS (윈도우 길이 누수)
# urine_missing: step7 SQL에서 생성 (소변 기록 없음 = 도뇨관 미삽입 등)
FEAT_MISSING_A = [
    "cr_missing",
    "lactate_missing",
    "urine_missing",   # step7 SQL 실행 후 생성
]


# ─────────────────────────────────────────────────────────────────────────────
# Track B  —  허혈성 피처
# ─────────────────────────────────────────────────────────────────────────────
# isch_map_min·isch_map_mean: vital_map_min·vital_map_mean 역할 겸임
# isch_shock_index: vital_shock_index 역할 겸임
FEAT_ISCHEMIC = [
    "isch_map_mean", "isch_map_min",
    "map_below65_hours", "flag_ischemia_over120min",
    "isch_shock_index",
    "vasopressor_flag",
]

# hemoglobin_missing·shock_index_missing 제거 → EXCLUDE_COLS (윈도우 길이 누수)
FEAT_MISSING_B = [
    "isch_lactate_missing",
]


# ─────────────────────────────────────────────────────────────────────────────
# Track C  —  신독성 약물
# ─────────────────────────────────────────────────────────────────────────────
FEAT_DRUG = [
    "vancomycin_rx", "vancomycin_exposure_hours",
    "piptazo_rx", "aminoglycoside_rx", "amphotericin_b_rx", "carbapenem_rx",
    "ketorolac_rx", "nsaid_any_rx",
    "ace_inhibitor_rx", "arb_rx", "acei_arb_any_rx",
    "furosemide_rx", "furosemide_cumulative_mg",
    "tacrolimus_rx", "cyclosporine_rx",
    "metformin_rx", "ppi_rx",
    "vanco_piptazo_combo",
    "vanco_aminogly_combo",
    "vanco_carbapenem_combo",
    "nsaid_acei_combo",
    "triple_whammy",
    "diuretic_overload_flag",
    "nephrotoxic_burden_score",
    "drug_risk_score",
]


# ─────────────────────────────────────────────────────────────────────────────
# SCR-06  —  규칙 기반 점수
# ─────────────────────────────────────────────────────────────────────────────
FEAT_RULE = [
    "score_cr", "score_bun", "score_egfr", "score_ischemia", "score_map",
    "rule_based_score",
    "total_nephrotoxic_burden",
]


# ─────────────────────────────────────────────────────────────────────────────
# 인구통계
# ─────────────────────────────────────────────────────────────────────────────
FEAT_DEMO = [
    "age",
    "gender",
    "first_careunit",
    # icu_los_hours 제거 → EXCLUDE_COLS (미래 정보)
]


# ─────────────────────────────────────────────────────────────────────────────
# 전체 피처 통합
# ─────────────────────────────────────────────────────────────────────────────
ALL_FEATURES: list[str] = (
    FEAT_LAB
    + FEAT_VITAL
    + FEAT_URINE
    + FEAT_MISSING_A
    + FEAT_ISCHEMIC
    + FEAT_MISSING_B
    + FEAT_DRUG
    + FEAT_RULE
    + FEAT_DEMO
)

FEAT_NLP_PRESET = [
    "kw_oliguria", "kw_anuria", "kw_edema", "kw_hydronephrosis",
    "kw_aki_mention", "kw_renal_abnormal", "kw_fluid_overload",
    "kw_pulmonary_edema", "kw_pleural_effusion", "kw_ascites",
    "kw_contrast_agent", "kw_rad_hydronephrosis", "kw_renal_calculus",
    "kw_cardiomegaly", "kw_foley_catheter", "kw_rad_aki_mention",
    "nlp_keyword_score", "rad_report_count",
    "nlp_missing", "rad_text_missing",
    "nlp_direct_renal_flag", "nlp_fluid_burden_flag",
]
ALL_FEATURES += FEAT_NLP_PRESET

FEAT_CAT: list[str] = ["gender", "first_careunit"]


# ─────────────────────────────────────────────────────────────────────────────
# 이상치 클리핑 범위
# ─────────────────────────────────────────────────────────────────────────────
CLIP_RULES: dict[str, tuple[float, float]] = {
    # 혈액검사
    "cr_min":                    (0,     30),
    "cr_max":                    (0,     30),
    "cr_delta":                  (0,     20),
    "bun_max":                   (0,    300),
    "bun_cr_ratio":              (0,    100),
    "potassium_max":             (1,     10),
    "bicarbonate_min":           (0,     45),
    "hemoglobin_min":            (0,     25),
    "lactate_max":               (0,     30),
    "egfr_ckdepi":               (0,    200),
    # ★ 활력징후 (step7 SQL 추가 후 실제 클리핑 적용)
    "hr_min":                    (20,   250),
    "hr_max":                    (20,   250),
    "hr_mean":                   (20,   250),
    "rr_max":                    (4,     60),
    "rr_mean":                   (4,     60),
    "sbp_min":                   (40,   300),
    "sbp_mean":                  (40,   300),
    "temp_min":                  (25,    45),  # °C
    "temp_mean":                 (25,    45),
    "spo2_min":                  (50,   100),
    "spo2_mean":                 (50,   100),
    # 허혈성 피처
    "map_below65_hours":         (0,     72),
    "isch_shock_index":          (0,      5),
    # 약물
    "vancomycin_exposure_hours": (0,    720),
    "furosemide_cumulative_mg":  (0,   5000),
    # 규칙 점수
    "rule_based_score":          (0,    100),
    "total_nephrotoxic_burden":  (0,     20),
    # ★ 소변량·수액 균형 (step7 SQL 추가 후 실제 클리핑 적용)
    "urine_6h":                  (0,   2000),  # mL
    "urine_24h":                 (0,   8000),  # mL
    "urine_total":               (0,  20000),  # mL
    "urine_zero_ratio":          (0,      1),  # 비율
    "fluid_24h":                 (0,  15000),  # mL
    "fluid_total":               (0,  50000),  # mL
    "fluid_balance_24h":         (-5000,  15000),
    "fluid_balance_total":       (-10000, 50000),
}


# ─────────────────────────────────────────────────────────────────────────────
# 범주형 인코딩 고정 매핑
# ─────────────────────────────────────────────────────────────────────────────
GENDER_MAP: dict[str, int] = {"M": 0, "F": 1, "Unknown": -1}

CAREUNIT_MAP: dict[str, int] = {
    "Medical Intensive Care Unit (MICU)":                0,
    "Surgical Intensive Care Unit (SICU)":               1,
    "Cardiac Vascular Intensive Care Unit (CVICU)":      2,
    "Medical/Surgical Intensive Care Unit (MICU/SICU)":  3,
    "Coronary Care Unit (CCU)":                          4,
    "Neuro Surgical Intensive Care Unit (Neuro SICU)":   5,
    "Trauma SICU (TSICU)":                               6,
    "Neuro Intermediate":                                7,
    "Unknown":                                          -1,
}

STATIC_ENCODERS: dict[str, dict[str, int]] = {
    "gender":        GENDER_MAP,
    "first_careunit":CAREUNIT_MAP,
}
	
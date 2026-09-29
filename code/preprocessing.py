"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
xgb_model/preprocessing.py  —  학습·추론 공용 전처리 파이프라인 (경로 수정)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
  데이터프레임을 XGBoost 입력 형태로 변환한다.
  학습(train.py)과 추론(inference.py) 양쪽에서 동일 함수를 호출해
  전처리 불일치(Training-Serving Skew)를 원천 차단한다.

전처리 단계:
  1. 경쟁 위험·재원 단기 환자 제외 (학습 전용)
  2. 사용할 피처 컬럼 선택
  3. 범주형 인코딩  — LabelEncoder pickle 또는 고정 딕셔너리
  4. 이상치 클리핑 — CLIP_RULES 범위 강제
  5. 피처 컬럼 순서 정렬 — 학습 순서와 추론 순서 일치 보장

저장 아티팩트 (train.py에서 생성):
  xgb_model/model/label_encoders.pkl   LabelEncoder 인스턴스 딕셔너리
  xgb_model/model/feature_names.csv    학습에 사용된 최종 피처 목록 (순서 포함)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import pickle
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from feature_config import (
    ALL_FEATURES, FEAT_CAT, CLIP_RULES,
    STATIC_ENCODERS, EXCLUDE_COLS, TARGET,
)

warnings.filterwarnings("ignore")

# ── 경로 설정 (절대경로) ────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_BASE_DIR, "model")

ENCODER_PATH       = os.path.join(_MODEL_DIR, "label_encoders.pkl")
FEATURE_NAMES_PATH = os.path.join(_MODEL_DIR, "feature_names.csv")

print(f"[preprocessing] BASE_DIR: {_BASE_DIR}")
print(f"[preprocessing] MODEL_DIR: {_MODEL_DIR}")
print(f"[preprocessing] ENCODER_PATH: {ENCODER_PATH}")
print(f"[preprocessing] FEATURE_NAMES_PATH: {FEATURE_NAMES_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. 코호트 필터링 (학습 전용)
# ─────────────────────────────────────────────────────────────────────────────

def filter_training_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """학습 데이터에서 신뢰할 수 없는 샘플을 제거한다.

    제거 대상:
      - icu_los_hours < 24  : 재원 24h 미만 → 피처 대부분 NULL
      - competed_with_death : 사망이 AKI보다 먼저 발생 → aki_label=0이 오분류

    Args:
        df: cdss_master_features 로드 원본 DataFrame

    Returns:
        필터링된 DataFrame (인덱스 리셋)
    """
    n_raw = len(df)

    # 재원 24h 미만 제거
    # ★ 주의: icu_los_hours는 피처에서 제외됐지만(EXCLUDE_COLS),
    #   코호트 필터링 기준으로는 여전히 사용한다. (필터 후 컬럼 자체는 drop)
    if "icu_los_hours" in df.columns:
        n_short = (df["icu_los_hours"] < 24).sum()
        df = df[df["icu_los_hours"] >= 24].copy()
    else:
        n_short = 0

    # 경쟁 위험 환자 제거
    n_competing = 0
    if "competed_with_death" in df.columns:
        n_competing = (df["competed_with_death"] == 1).sum()
        df = df[df["competed_with_death"] == 0].copy()

    n_final = len(df)
    print(f"[코호트 필터] 전체: {n_raw:,}  →  재원단기 -{n_short:,}  경쟁위험 -{n_competing:,}")
    print(f"             최종: {n_final:,}행  AKI 비율: {df[TARGET].mean()*100:.1f}%")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 범주형 인코딩
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_save_label_encoders(df: pd.DataFrame) -> dict[str, LabelEncoder]:
    """학습 데이터에서 LabelEncoder를 fit하고 pickle로 저장한다."""
    encoders: dict[str, LabelEncoder] = {}
    for col in FEAT_CAT:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        le.fit(df[col].fillna("Unknown").astype(str))
        encoders[col] = le
        print(f"  [인코더 학습] {col}: {list(le.classes_)}")

    os.makedirs(_MODEL_DIR, exist_ok=True)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoders, f)
    print(f"  [저장] {ENCODER_PATH}")
    return encoders


def load_label_encoders() -> Optional[dict[str, LabelEncoder]]:
    """저장된 LabelEncoder pickle 파일을 로드한다."""
    if not os.path.exists(ENCODER_PATH):
        print(f"[경고] 인코더 파일 없음: {ENCODER_PATH}")
        return None

    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)
        print(f"[로드] 인코더 로드 완료: {ENCODER_PATH}")
        return encoders


def encode_categorical_columns(
    df: pd.DataFrame,
    encoders: Optional[dict[str, LabelEncoder]] = None,
) -> pd.DataFrame:
    """범주형 컬럼을 정수로 인코딩한다."""
    df = df.copy()
    for col in FEAT_CAT:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("Unknown").astype(str)

        if encoders and col in encoders:
            enc = encoders[col]
            df[col] = df[col].map(
                lambda v, e=enc: int(e.transform([v])[0])
                if v in e.classes_ else -1
            )
        else:
            mapping = STATIC_ENCODERS.get(col, {})
            df[col] = df[col].map(lambda v, m=mapping: m.get(v, -1))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. 이상치 클리핑
# ─────────────────────────────────────────────────────────────────────────────

def clip_outliers_by_clinical_range(df: pd.DataFrame) -> pd.DataFrame:
    """생리적으로 불가능하거나 측정 오류인 값을 CLIP_RULES 범위로 강제 조정한다."""
    df = df.copy()
    for col, (lo, hi) in CLIP_RULES.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. 피처 선택 및 컬럼 정렬
# ─────────────────────────────────────────────────────────────────────────────

def select_and_order_features(
    df: pd.DataFrame,
    feature_names: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """DataFrame에서 학습에 사용할 피처만 선택하고 순서를 정렬한다.

    feature_names가 주어지면 그 순서를 따른다. (추론 시 필수)
    없으면 ALL_FEATURES에서 실제로 존재하는 컬럼만 사용한다. (학습 시)

    Args:
        df:            입력 DataFrame
        feature_names: 사용할 피처 이름 목록 (순서 포함)

    Returns:
        (선택된 X DataFrame, 실제 사용된 feature_names 리스트)
    """
    if feature_names is None:
        # ★ 수정: EXCLUDE_COLS를 이중 안전망으로 적용
        # ALL_FEATURES에 실수로 누수 컬럼이 추가돼도 여기서 차단된다.
        available = [
            f for f in ALL_FEATURES
            if f in df.columns and f not in EXCLUDE_COLS
        ]

        # 디버그 출력: DB에 없는 피처
        missing_in_db = [f for f in ALL_FEATURES if f not in df.columns]
        if missing_in_db:
            print(
                f"  [경고] DB에 없는 피처 {len(missing_in_db)}개 제외: "
                f"{missing_in_db[:5]}{'...' if len(missing_in_db) > 5 else ''}"
            )

        # 디버그 출력: EXCLUDE_COLS에 걸려 차단된 피처
        blocked_by_exclude = [
            f for f in ALL_FEATURES
            if f in df.columns and f in EXCLUDE_COLS
        ]
        if blocked_by_exclude:
            print(
                f"  [안전망] EXCLUDE_COLS로 차단된 피처 {len(blocked_by_exclude)}개: "
                f"{blocked_by_exclude}"
            )

        feature_names = available
    else:
        # 추론: 지정된 피처 순서 그대로. 없는 컬럼은 NaN으로 채움
        for col in feature_names:
            if col not in df.columns:
                df[col] = np.nan

    X = df[feature_names].copy()
    return X, feature_names


def save_feature_names(feature_names: list[str]) -> None:
    """학습에 사용된 피처 목록을 CSV로 저장한다."""
    os.makedirs(_MODEL_DIR, exist_ok=True)
    pd.DataFrame(feature_names).to_csv(
        FEATURE_NAMES_PATH,
        index=False,
        header=False,
    )
    print(f"  [저장] {FEATURE_NAMES_PATH}  ({len(feature_names)}개 피처)")


def load_feature_names() -> Optional[list[str]]:
    """저장된 피처 목록 CSV를 로드한다."""
    if not os.path.exists(FEATURE_NAMES_PATH):
        print(f"[경고] 피처 목록 파일 없음: {FEATURE_NAMES_PATH}")
        return None

    df = pd.read_csv(FEATURE_NAMES_PATH, header=None)
    feature_names = df[0].tolist()
    print(f"[로드] 피처 목록 로드 완료: {len(feature_names)}개 ({FEATURE_NAMES_PATH})")
    return feature_names


# ─────────────────────────────────────────────────────────────────────────────
# 5. 통합 전처리 함수
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_training(
    df: pd.DataFrame,
    return_meta: bool = False,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, LabelEncoder]] | tuple[pd.DataFrame, pd.Series, list[str], dict[str, LabelEncoder], pd.DataFrame]:
    """학습용 전처리 통합 함수.

    filter → encode → clip → select 순서로 실행한다.
    인코더와 피처 목록을 xgb_model/model/ 디렉터리에 저장한다.

    Args:
        df: cdss_master_features 로드 원본

    Returns:
        (X, y, feature_names, encoders)
    """
    print("\\n[전처리 시작 - 학습 모드]")

    # 1. 코호트 필터링
    df = filter_training_cohort(df)

    # 2. 타겟 분리
    y = df[TARGET].astype(int)
    meta_df = df[[c for c in ["subject_id", "stay_id"] if c in df.columns]].copy()

    # 3. 범주형 인코딩 (fit + save)
    encoders = fit_and_save_label_encoders(df)
    df = encode_categorical_columns(df, encoders)

    # 4. 이상치 클리핑
    df = clip_outliers_by_clinical_range(df)

    # 5. 피처 선택·정렬 + 저장
    X, feature_names = select_and_order_features(df)
    save_feature_names(feature_names)

    print(f"[전처리 완료] X: {X.shape}  y 양성률: {y.mean()*100:.1f}%\\n")
    if return_meta:
        return X, y, feature_names, encoders, meta_df.reset_index(drop=True)
    return X, y, feature_names, encoders


def preprocess_for_inference(
    df: pd.DataFrame,
    feature_names: list[str],
    encoders: Optional[dict[str, LabelEncoder]] = None,
) -> pd.DataFrame:
    """추론용 전처리 통합 함수.

    학습 때와 완전히 동일한 순서·방식으로 전처리한다.
    필터링은 하지 않는다 (추론 대상은 모두 현재 ICU 입원 중인 환자).

    Args:
        df:            단일 환자 또는 배치 DataFrame
        feature_names: 학습 시 저장된 피처 목록 (load_feature_names()로 로드)
        encoders:      학습 시 저장된 LabelEncoder (load_label_encoders()로 로드)

    Returns:
        XGBoost 입력 준비된 X DataFrame (feature_names 순서)
    """
    if encoders is None:
        encoders = load_label_encoders()

    df = encode_categorical_columns(df, encoders)
    df = clip_outliers_by_clinical_range(df)
    X, _ = select_and_order_features(df, feature_names)
    return X
	
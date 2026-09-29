"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
xgb_model/train.py  —  XGBoost AKI 예측 모델 학습 파이프라인 (SHAP 포함)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[개선 사항]
  ① F1 / Precision / Recall 지표 추가  (Youden 임계값 기준)
  ② 수치형 결측 → missing_flag 보존 후 KNN Imputation
  ③ map_missing 외 주요 변수 결측 패턴을 피처로 추가
  ④ ★ KNN Imputer를 CV 루프 안으로 이동 (누수 방지)
     Fold별로 train만 fit_transform, val은 transform만 적용한다.
  ⑤ ★★ KNN Imputer를 Optuna objective 루프 안으로도 이동 (누수 완전 차단)
     기존: 전체 X를 impute 후 Optuna에 전달 → val 정보 혼입
     수정: objective 내부 각 fold에서 train만 fit → val에 transform

[전체 파이프라인 흐름]
  DB 로드 → missing_flag 생성 → 전처리 → Optuna 탐색(fold별 imputation)
  → 5-Fold CV(raw X, fold별 imputation) → 최종 모델 학습(전체 imputed)
  → 아티팩트 저장 → SHAP 분석

실행 방법:
    python train.py                        # 기본 실행
    python train.py --trials 100           # Optuna 시도 횟수 지정
    python train.py --db-uri postgresql://user:pw@localhost:5432/mimic4

출력 아티팩트:
    model/xgb_aki.json          XGBoost 모델 가중치
    model/threshold.txt         최적 분류 임계값 (Youden's J)
    model/feature_names.csv     학습에 사용된 피처 목록 (순서 포함)
    model/label_encoders.pkl    LabelEncoder 인스턴스 딕셔너리
    model/knn_imputer.pkl       KNNImputer 인스턴스 (추론 시 재사용)
    output/eval_metrics.txt     5-Fold CV 성능 지표 (F1/Precision/Recall 포함)
    output/feature_importance.csv  XGBoost feature importance
    output/track_importance.csv    트랙별 기여도 (SCR-03~07 그룹)
    output/shap_summary_plot.png   SHAP Beeswarm 플롯
    output/shap_summary_bar.png    SHAP 막대 그래프 (피처 기여도)
    output/shap_values.npy         SHAP values (numpy)
    output/shap_base_values.npy    SHAP base values (numpy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import pickle
import argparse
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import sqlalchemy
import xgboost as xgb
import optuna
import shap
import matplotlib.pyplot as plt

from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.impute import KNNImputer

from feature_config import (
    ALL_FEATURES,
    FEAT_LAB,
    FEAT_ISCHEMIC,
    FEAT_DRUG,
    FEAT_RULE,
    TARGET,
)
from preprocessing import preprocess_for_training

warnings.filterwarnings("ignore")

os.makedirs("model",  exist_ok=True)
os.makedirs("output", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 전역 상수 / 설정값
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://bio4:bio4@localhost:5432/mimic4"
)

N_FOLDS         = 5
N_OPTUNA_TRIALS = 20
EARLY_STOP      = 30
RANDOM_STATE    = 42
KNN_NEIGHBORS   = 5

MISSING_FLAG_COLS = [
    "map",
    "hemoglobin",
    "potassium",
    "creatinine",
    "urine_output",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_master_features_from_db(db_uri: str) -> pd.DataFrame:
    print("[데이터 로드] cdss_master_features ...")
    engine = sqlalchemy.create_engine(db_uri)
    df = pd.read_sql("SELECT * FROM cdss_master_features", engine)
    print(f"  로드 완료: {len(df):,}행 × {df.shape[1]}열")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. 결측 플래그 생성
# ─────────────────────────────────────────────────────────────────────────────

def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """결측 패턴이 임상적 의미를 갖는 컬럼에 대해 이진 _missing 플래그를 추가한다.

    반드시 KNN imputation 이전에 호출해야 한다.
    """
    df = df.copy()
    added = []

    for col in MISSING_FLAG_COLS:
        flag_col = f"{col}_missing"
        if col in df.columns and flag_col not in df.columns:
            df[flag_col] = df[col].isna().astype(np.int8)
            added.append(flag_col)

    if added:
        print(f"  [missing_flag] 추가된 플래그 피처: {added}")
    else:
        print("  [missing_flag] 새로 추가할 플래그 없음 (이미 모두 존재)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. KNN Imputation (전체 데이터용 — 최종 모델 학습에만 사용)
# ─────────────────────────────────────────────────────────────────────────────

def fit_knn_imputer(X: pd.DataFrame) -> tuple[pd.DataFrame, KNNImputer]:
    """수치형 컬럼의 결측값을 KNN Imputation으로 채우고, Imputer를 저장한다.

    [사용 시점]
    - 최종 모델 학습용: 전체 데이터로 fit_transform 후 저장
    - Optuna 탐색용 / CV 평가용: ★ 이 함수를 쓰지 않고
      각 루프(fold) 안에서 train만으로 fit → val에 transform만 적용 (누수 방지)

    Args:
        X: 전처리된 피처 DataFrame

    Returns:
        (imputed_X, fitted_imputer)
    """
    numeric_cols = [
        c for c in X.select_dtypes(include=[np.number]).columns
        if not c.endswith("_missing")
    ]
    non_numeric_cols = [c for c in X.columns if c not in numeric_cols]

    print(f"\\n[KNN Imputation] 수치형 {len(numeric_cols)}개 컬럼 대상 (k={KNN_NEIGHBORS})")
    before_missing = X[numeric_cols].isna().sum().sum()
    print(f"  imputation 전 총 결측 셀: {before_missing:,}")

    imputer       = KNNImputer(n_neighbors=KNN_NEIGHBORS)
    X_imputed_arr = imputer.fit_transform(X[numeric_cols])

    X_imputed_num = pd.DataFrame(X_imputed_arr, columns=numeric_cols, index=X.index)

    after_missing = X_imputed_num.isna().sum().sum()
    print(f"  imputation 후 총 결측 셀: {after_missing:,}")

    X_result = pd.concat([X_imputed_num, X[non_numeric_cols]], axis=1)[X.columns]

    imputer_path = "model/knn_imputer.pkl"
    with open(imputer_path, "wb") as f:
        pickle.dump({"imputer": imputer, "numeric_cols": numeric_cols}, f)
    print(f"  [저장] {imputer_path}")

    return X_result, imputer


# ─────────────────────────────────────────────────────────────────────────────
# 4. Optuna 하이퍼파라미터 탐색
# ★ 수정: objective 내부 각 fold에서 KNN imputation 수행 (누수 완전 차단)
# ─────────────────────────────────────────────────────────────────────────────

def build_xgb_params_from_optuna_trial(trial: optuna.Trial, scale_pos_weight: float) -> dict:
    return {
        "max_depth":          trial.suggest_int("max_depth",           4,    8),
        "learning_rate":      trial.suggest_float("learning_rate",     0.01, 0.2,  log=True),
        "n_estimators":       trial.suggest_int("n_estimators",      100,  800),
        "subsample":          trial.suggest_float("subsample",         0.6,  1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree",  0.5,  1.0),
        "min_child_weight":   trial.suggest_int("min_child_weight",    1,   20),
        "gamma":              trial.suggest_float("gamma",             0.0,  5.0),
        "reg_alpha":          trial.suggest_float("reg_alpha",         1e-4, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda",        1e-4, 10.0, log=True),
        "scale_pos_weight":   scale_pos_weight,
        "tree_method":        "hist",
        "eval_metric":        "aucpr",
        "use_label_encoder":  False,
        "random_state":       RANDOM_STATE,
        "n_jobs":             -1,
    }


def run_optuna_hyperparameter_search(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series] = None,
    n_trials: int = N_OPTUNA_TRIALS,
) -> dict:
    """Optuna TPE 샘플러로 XGBoost 하이퍼파라미터를 탐색하고 최적값을 반환한다.

    ★ 수정 핵심:
    X는 imputation 전 원본 X_raw를 받는다.
    objective 내부 각 fold에서 train만으로 KNNImputer를 fit하고,
    val에는 transform만 적용해 val 정보가 train imputation에 혼입되지 않도록 한다.

    Args:
        X: ★ imputation 전 원본 X_raw
        y: AKI 타겟 Series
        n_trials: Optuna 탐색 횟수
    """
    neg_count        = (y == 0).sum()
    pos_count        = (y == 1).sum()
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    if groups is not None:
        cv_inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # ★ 수정: imputation 대상 컬럼 목록을 루프 밖에서 한 번만 계산 (고정)
    numeric_cols = [
        c for c in X.select_dtypes(include=[np.number]).columns
        if not c.endswith("_missing")
    ]

    def objective(trial: optuna.Trial) -> float:
        params    = build_xgb_params_from_optuna_trial(trial, scale_pos_weight)
        fold_aucs = []

        split_iter = cv_inner.split(X, y, groups) if groups is not None else cv_inner.split(X, y)
        for fold_i, (tr_idx, val_idx) in enumerate(split_iter):
            # ★ 수정: .copy()로 원본 X 보호
            X_tr  = X.iloc[tr_idx].copy()
            X_val = X.iloc[val_idx].copy()
            y_tr  = y.iloc[tr_idx]
            y_val = y.iloc[val_idx]

            # ★ 수정: train fold만으로 imputer fit → val fold는 transform만
            # 기존 코드: 전체 X_optuna를 미리 impute 후 fold로 쪼갬
            #           → val 정보가 train imputation에 혼입 (누수)
            # 수정 코드: fold로 쪼갠 후 train만으로 fit → val에 transform만
            #           → val 정보 완전 차단
            fold_imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
            X_tr[numeric_cols]  = fold_imputer.fit_transform(X_tr[numeric_cols])
            X_val[numeric_cols] = fold_imputer.transform(X_val[numeric_cols])

            model = xgb.XGBClassifier(**params, early_stopping_rounds=EARLY_STOP)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            prob = model.predict_proba(X_val)[:, 1]
            fold_aucs.append(roc_auc_score(y_val, prob))

            trial.report(np.mean(fold_aucs), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_aucs)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\\n[Optuna] {n_trials}회 탐색 시작 (fold별 KNN imputation 적용) ...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({
        "scale_pos_weight":  scale_pos_weight,
        "tree_method":       "hist",
        "eval_metric":       "aucpr",
        "use_label_encoder": False,
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
    })

    print(f"  최적 AUROC: {study.best_value:.4f}")
    print(f"  최적 파라미터: {best_params}")
    return best_params


# ─────────────────────────────────────────────────────────────────────────────
# 5. 5-Fold CV 성능 평가
# ★ fold별 KNN imputation (누수 방지) — 기존과 동일
# ─────────────────────────────────────────────────────────────────────────────

def run_stratified_kfold_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict,
    groups: Optional[pd.Series] = None,
) -> dict:
    """Optuna 최적 파라미터로 5-Fold CV를 수행해 최종 성능을 평가한다.

    KNN Imputer를 각 Fold의 train 데이터로만 fit하고,
    val 데이터에는 transform만 적용한다.

    Args:
        X: ★ imputation 전 원본 X_raw
        y: AKI 타겟 Series
        best_params: Optuna 최적 하이퍼파라미터
    """
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metrics = {k: [] for k in ["auroc", "auprc", "f1", "precision", "recall", "specificity", "threshold"]}

    numeric_cols = [
        c for c in X.select_dtypes(include=[np.number]).columns
        if not c.endswith("_missing")
    ]

    print(f"\\n[5-Fold CV] 최적 파라미터로 성능 평가 (fold별 KNN imputation) ...")
    split_iter = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for fold_i, (tr_idx, val_idx) in enumerate(split_iter, 1):

        X_tr_raw  = X.iloc[tr_idx].copy()
        X_val_raw = X.iloc[val_idx].copy()
        y_tr      = y.iloc[tr_idx]
        y_val     = y.iloc[val_idx]

        fold_imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
        X_tr_raw[numeric_cols]  = fold_imputer.fit_transform(X_tr_raw[numeric_cols])
        X_val_raw[numeric_cols] = fold_imputer.transform(X_val_raw[numeric_cols])

        model = xgb.XGBClassifier(**best_params, early_stopping_rounds=EARLY_STOP)
        model.fit(
            X_tr_raw, y_tr,
            eval_set=[(X_val_raw, y_val)],
            verbose=False,
        )

        prob      = model.predict_proba(X_val_raw)[:, 1]
        auroc     = roc_auc_score(y_val, prob)
        auprc     = average_precision_score(y_val, prob)
        threshold = _find_youden_threshold(y_val, prob)
        pred      = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
        specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1        = f1_score(y_val,        pred, zero_division=0)
        precision = precision_score(y_val, pred, zero_division=0)
        recall    = recall_score(y_val,    pred, zero_division=0)

        for key, val in zip(
            ["auroc", "auprc", "f1", "precision", "recall", "specificity", "threshold"],
            [auroc,   auprc,   f1,   precision,   recall,   specificity,   threshold],
        ):
            metrics[key].append(val)

        print(
            f"  Fold {fold_i}: "
            f"AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
            f"F1={f1:.4f}  Prec={precision:.4f}  Recall={recall:.4f}  Spec={specificity:.4f}  "
            f"Threshold={threshold:.3f}"
        )

    print(f"\\n  [CV 요약]")
    for key, label in [
        ("auroc",     "AUROC    "),
        ("auprc",     "AUPRC    "),
        ("f1",        "F1       "),
        ("precision", "Precision"),
        ("recall",    "Recall   "),
        ("specificity","Specificity"),
    ]:
        vals = metrics[key]
        print(f"  {label}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"  최적 임계값 평균: {np.mean(metrics['threshold']):.3f}")

    return metrics


def _find_youden_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """ROC 곡선에서 Youden's J 통계량이 최대가 되는 분류 임계값을 반환한다."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


# ─────────────────────────────────────────────────────────────────────────────
# 6. 최종 모델 학습 (전체 데이터)
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model_on_full_data(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict,
    best_threshold: float,
) -> xgb.XGBClassifier:
    """CV로 검증된 최적 파라미터로 전체 데이터에서 최종 모델을 학습한다."""
    print("\\n[최종 모델 학습] 전체 데이터로 재학습 ...")

    params = {k: v for k, v in best_params.items() if k != "early_stopping_rounds"}

    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)

    model_path = os.getenv("XGB_MODEL_PATH", "model/xgb_aki.json")
    model.save_model(model_path)
    print(f"  [저장] {model_path}")

    threshold_path = os.getenv("XGB_THRESHOLD_PATH", "model/threshold.txt")
    with open(threshold_path, "w", encoding="utf-8") as f:
        f.write(str(best_threshold))
    print(f"  [저장] {threshold_path}  (threshold={best_threshold:.3f})")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 7. Feature Importance 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_feature_importance_reports(
    model: xgb.XGBClassifier,
    feature_names: list[str],
) -> None:
    """XGBoost 피처 중요도를 개별 피처 단위·CDSS 트랙 단위로 CSV에 저장한다."""
    importance = model.get_booster().get_score(importance_type="gain")

    df_imp = pd.DataFrame([
        {"feature": f, "importance_gain": importance.get(f, 0.0)}
        for f in feature_names
    ]).sort_values("importance_gain", ascending=False)

    df_imp.to_csv("output/feature_importance.csv", index=False, encoding="utf-8")
    print("\\n  [저장] output/feature_importance.csv")
    print("  상위 10 피처:")
    print(df_imp.head(10).to_string(index=False))

    TRACK_GROUPS = {
        "SCR-03 약물":     FEAT_DRUG,
        "SCR-04 혈액검사": FEAT_LAB,
        "SCR-05 허혈":     FEAT_ISCHEMIC,
        "SCR-06 규칙":     FEAT_RULE,
    }
    track_rows = []
    for track_name, feat_list in TRACK_GROUPS.items():
        total = sum(importance.get(f, 0.0) for f in feat_list)
        track_rows.append({"track": track_name, "total_gain": round(total, 2)})

    df_track = pd.DataFrame(track_rows).sort_values("total_gain", ascending=False)
    df_track["pct"] = (df_track["total_gain"] / df_track["total_gain"].sum() * 100).round(1)
    df_track.to_csv("output/track_importance.csv", index=False, encoding="utf-8")
    print("\\n  [저장] output/track_importance.csv")
    print(df_track.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 8. 평가 지표 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_evaluation_metrics(cv_metrics: dict, best_params: dict) -> None:
    """5-Fold CV 성능 지표를 텍스트 파일에 저장한다."""
    label_map = {
        "auroc":     "AUROC    ",
        "auprc":     "AUPRC    ",
        "f1":        "F1       ",
        "precision": "Precision",
        "recall":    "Recall   ",
        "specificity":"Specificity",
        "threshold": "Threshold",
    }

    lines = ["=== AKI XGBoost 5-Fold CV 결과 ===\\n"]
    for key, label in label_map.items():
        vals = cv_metrics[key]
        lines.append(f"{label}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    lines += ["", "=== 최적 하이퍼파라미터 ==="]
    lines += [f"  {k}: {v}" for k, v in best_params.items()]

    with open("output/eval_metrics.txt", "w", encoding="utf-8") as f:
        f.write("\\n".join(lines))
    print("\\n  [저장] output/eval_metrics.txt")


def split_train_test_by_subject(
    X: pd.DataFrame,
    y: pd.Series,
    subject_ids: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """subject_id 기준으로 누수 없이 train/test 8:2 분할한다."""
    df_split = pd.DataFrame({"subject_id": subject_ids.astype(str), "y": y.values})
    subj_label = df_split.groupby("subject_id")["y"].max().reset_index()

    tr_subj, te_subj = train_test_split(
        subj_label["subject_id"],
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=subj_label["y"],
    )

    tr_mask = subject_ids.astype(str).isin(set(tr_subj))
    te_mask = subject_ids.astype(str).isin(set(te_subj))

    X_tr, X_te = X.loc[tr_mask].reset_index(drop=True), X.loc[te_mask].reset_index(drop=True)
    y_tr, y_te = y.loc[tr_mask].reset_index(drop=True), y.loc[te_mask].reset_index(drop=True)
    g_tr, g_te = subject_ids.loc[tr_mask].reset_index(drop=True), subject_ids.loc[te_mask].reset_index(drop=True)

    print(
        f"[Split] train={len(X_tr):,} ({y_tr.mean()*100:.1f}% pos), "
        f"test={len(X_te):,} ({y_te.mean()*100:.1f}% pos), "
        f"test_size={test_size:.1%}"
    )
    print(f"        unique_subject train={g_tr.nunique():,}, test={g_te.nunique():,}")
    return X_tr, X_te, y_tr, y_te, g_tr, g_te


def evaluate_on_holdout_test(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict:
    """홀드아웃 test 성능을 계산하고 저장한다."""
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics = {
        "auroc": roc_auc_score(y_test, prob),
        "auprc": average_precision_score(y_test, prob),
        "f1": f1_score(y_test, pred, zero_division=0),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "specificity": specificity,
        "threshold": threshold,
        "n_test": int(len(y_test)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    out_path = "output/test_metrics.txt"
    lines = ["=== Hold-out Test Metrics ==="] + [f"{k}: {v}" for k, v in metrics.items()]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\\n".join(lines))
    print(f"\\n[Test] AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
          f"F1={metrics['f1']:.4f} Prec={metrics['precision']:.4f} "
          f"Recall={metrics['recall']:.4f} Spec={metrics['specificity']:.4f}")
    print(f"  [저장] {out_path}")
    return metrics


def save_markdown_report(
    cv_metrics: dict,
    test_metrics: dict,
    best_params: dict,
    best_threshold: float,
) -> None:
    """학습/평가/SHAP 요약을 markdown으로 저장한다."""
    md_path = "output/model_report.md"
    lines = []
    lines.append("# AKI XGBoost 모델 리포트")
    lines.append("")
    lines.append("## 1) CV 성능 (Train, 5-Fold Group CV)")
    for key, label in [
        ("auroc", "ROC-AUC"),
        ("auprc", "PR-AUC"),
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "민감도(Recall)"),
        ("specificity", "특이도(Specificity)"),
        ("threshold", "Threshold"),
    ]:
        vals = cv_metrics[key]
        lines.append(f"- {label}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    lines.append("")
    lines.append("## 2) Hold-out Test 성능 (8:2)")
    lines.append(f"- ROC-AUC: {test_metrics['auroc']:.4f}")
    lines.append(f"- PR-AUC: {test_metrics['auprc']:.4f}")
    lines.append(f"- Threshold: {test_metrics['threshold']:.4f}")
    lines.append(f"- 민감도(Recall): {test_metrics['recall']:.4f}")
    lines.append(f"- 특이도(Specificity): {test_metrics['specificity']:.4f}")
    lines.append(f"- Precision: {test_metrics['precision']:.4f}")
    lines.append(f"- F1: {test_metrics['f1']:.4f}")
    lines.append("")
    lines.append("### 혼동행렬 (Test)")
    lines.append("")
    lines.append("|              | Pred 0 | Pred 1 |")
    lines.append("|--------------|-------:|-------:|")
    lines.append(f"| True 0       | {test_metrics['tn']} | {test_metrics['fp']} |")
    lines.append(f"| True 1       | {test_metrics['fn']} | {test_metrics['tp']} |")

    lines.append("")
    lines.append("## 3) 최적 임계값")
    lines.append(f"- CV 평균 Youden threshold: {best_threshold:.4f}")

    lines.append("")
    lines.append("## 4) SHAP 결과")
    lines.append("- Summary plot:")
    lines.append("![SHAP Summary](./shap_summary_plot.png)")
    lines.append("- Importance bar:")
    lines.append("![SHAP Bar](./shap_summary_bar.png)")
    lines.append("")
    lines.append("상위 SHAP 피처는 `output/feature_importance.csv` 및 SHAP 이미지로 확인하세요.")

    lines.append("")
    lines.append("## 5) 최적 하이퍼파라미터")
    for k, v in best_params.items():
        lines.append(f"- {k}: {v}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\\n".join(lines))
    print(f"  [저장] {md_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SHAP 분석
# ─────────────────────────────────────────────────────────────────────────────

def save_shap_analysis(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    feature_names: list[str],
) -> None:
    """SHAP TreeExplainer로 모델 해석성을 분석하고 플롯·수치 파일을 저장한다."""
    print("\\n[SHAP 분석] TreeExplainer로 SHAP values 계산 중 ...")

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        base_value  = explainer.expected_value

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            base_value  = base_value[1] if isinstance(base_value, (list, np.ndarray)) else base_value

        print("  [1/3] SHAP Summary Plot (Beeswarm) 생성 중 ...")
        plt.figure(figsize=(14, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="dot", show=False)
        plt.tight_layout()
        plt.savefig("output/shap_summary_plot.png", dpi=300, bbox_inches="tight")
        print("    [저장] output/shap_summary_plot.png")
        plt.close()

        print("  [2/3] SHAP Feature Importance Bar 생성 중 ...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig("output/shap_summary_bar.png", dpi=300, bbox_inches="tight")
        print("    [저장] output/shap_summary_bar.png")
        plt.close()

        print("  [3/3] SHAP values 저장 중 ...")
        np.save("output/shap_values.npy",      shap_values)
        np.save("output/shap_base_values.npy", np.array(base_value))

        meta = {
            "feature_names":     feature_names,
            "shap_values_shape": shap_values.shape,
            "base_value":        float(base_value) if np.isscalar(base_value)
                                 else float(np.mean(base_value)),
        }
        np.save("output/shap_meta.npy", meta, allow_pickle=True)
        print("    [저장] output/shap_values.npy / shap_base_values.npy / shap_meta.npy")

        print("\\n  [SHAP 통계] 상위 10 중요 피처:")
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features  = np.argsort(mean_abs_shap)[-10:][::-1]
        for rank, idx in enumerate(top_features, 1):
            feat     = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            shap_imp = mean_abs_shap[idx]
            print(f"    {rank:2d}. {feat:30s} → SHAP importance: {shap_imp:.4f}")

    except Exception as e:
        print(f"\\n  ⚠️  SHAP 분석 중 오류 발생: {e}")
        print("  계속 진행합니다...")


# ─────────────────────────────────────────────────────────────────────────────
# 메인 실행
# ★ 수정: X_optuna 제거 — Optuna / CV 모두 X_raw 직접 전달
# ─────────────────────────────────────────────────────────────────────────────

def main(db_uri: str, n_trials: int) -> None:
    """XGBoost AKI 예측 모델의 전체 학습 파이프라인을 실행한다.

    ★ 수정된 파이프라인 흐름:
      1. DB 로드
      2. missing_flag 생성
      3. 전처리 (인코딩·클리핑·피처 선택)  → X_raw (imputation 전)
      4. Optuna 탐색: X_raw 전달 → objective 내부 fold별 imputation (누수 완전 차단)
      5. 5-Fold CV:  X_raw 전달 → 루프 안에서 fold별 imputation (누수 완전 차단)
      6. 최종 모델: X_raw를 전체 imputation한 X_final 사용 + imputer 저장
      7. 리포트 저장
      8. SHAP 분석

    [기존 vs 수정 비교]
      기존: fit_knn_imputer(X_raw) → X_optuna → Optuna(X_optuna)
            → val 정보가 train imputation에 혼입 (누수)
      수정: Optuna(X_raw) → objective 내부에서 fold별 fit/transform
            → val 정보 완전 차단
    """
    print("=" * 70)
    print("AKI CDSS XGBoost 학습 파이프라인 v4 (Optuna + CV 모두 fold별 KNN)")
    print("=" * 70)

    # 1. DB 로드
    df_raw = load_master_features_from_db(db_uri)

    # 2. missing_flag 생성 (imputation 전에 반드시 실행)
    df_flagged = add_missing_flags(df_raw)

    # 3. 전처리: 인코딩·클리핑·피처 선택 (+ subject_id 메타)
    X_raw, y, feature_names, encoders, meta_df = preprocess_for_training(df_flagged, return_meta=True)
    if "subject_id" not in meta_df.columns:
        raise ValueError("subject_id 컬럼이 없어 그룹 분할을 수행할 수 없습니다.")
    subject_ids = meta_df["subject_id"]

    # 4. subject_id 기준 train/test 8:2 고정 분할
    X_train_raw, X_test_raw, y_train, y_test, g_train, g_test = split_train_test_by_subject(
        X_raw, y, subject_ids, test_size=0.2
    )

    # 5. Optuna 탐색: train만 사용 + subject_id group CV
    best_params = run_optuna_hyperparameter_search(X_train_raw, y_train, groups=g_train, n_trials=n_trials)

    # 6. 5-Fold CV: train만 사용 + subject_id group CV
    print("\\n[CV] X_train_raw(imputation 전)로 fold별 KNN imputation 수행 ...")
    cv_metrics = run_stratified_kfold_cross_validation(X_train_raw, y_train, best_params, groups=g_train)
    best_threshold = float(np.mean(cv_metrics["threshold"]))

    # 7. 최종 모델: train 데이터로 imputation 후 학습 + imputer 저장
    print("\\n[최종 모델용 imputation] train 데이터로 fit_transform + imputer 저장 ...")
    X_train_final, knn_imputer = fit_knn_imputer(X_train_raw.copy())

    # feature_names: imputation 후 컬럼 순서 기준으로 갱신
    feature_names = list(X_train_final.columns)

    final_model = train_final_model_on_full_data(X_train_final, y_train, best_params, best_threshold)

    # 8. hold-out test 평가: train imputer로 test transform
    numeric_cols = [c for c in X_train_raw.select_dtypes(include=[np.number]).columns if not c.endswith("_missing")]
    with open("model/knn_imputer.pkl", "rb") as f:
        saved = pickle.load(f)
    imputer = saved["imputer"]
    X_test_final = X_test_raw.copy()
    X_test_final[numeric_cols] = imputer.transform(X_test_final[numeric_cols])
    test_metrics = evaluate_on_holdout_test(final_model, X_test_final, y_test, best_threshold)

    # 9. 리포트 저장
    save_feature_importance_reports(final_model, feature_names)
    save_evaluation_metrics(cv_metrics, best_params)
    save_markdown_report(cv_metrics, test_metrics, best_params, best_threshold)

    # 10. SHAP 분석 (train 기준)
    save_shap_analysis(final_model, X_train_final, feature_names)

    # 최종 아티팩트 확인
    print("\\n" + "=" * 70)
    print("학습 완료. 생성된 아티팩트:")
    artifact_paths = [
        "model/xgb_aki.json",
        "model/threshold.txt",
        "model/feature_names.csv",
        "model/label_encoders.pkl",
        "model/knn_imputer.pkl",
        "output/eval_metrics.txt",
        "output/test_metrics.txt",
        "output/feature_importance.csv",
        "output/track_importance.csv",
        "output/shap_summary_plot.png",
        "output/shap_summary_bar.png",
        "output/shap_values.npy",
        "output/shap_base_values.npy",
    ]
    for path in artifact_paths:
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {exists} {path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AKI XGBoost 학습 파이프라인 v4 (Optuna + CV 모두 fold별 KNN)"
    )
    parser.add_argument(
        "--db-uri",
        default=DEFAULT_DB_URI,
        help="SQLAlchemy DB URI",
    )
    parser.add_argument(
        "--trials",
        default=N_OPTUNA_TRIALS,
        type=int,
        help=f"Optuna 탐색 횟수 (기본값: {N_OPTUNA_TRIALS})",
    )
    args = parser.parse_args()
    main(db_uri=args.db_uri, n_trials=args.trials)
	
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
xgb_model/evaluate.py  —  모델 성능 평가 & 캘리브레이션
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
  학습 완료 모델의 임상적 유용성을 다각도로 평가한다.
  단순 AUROC 외에도 AKI 예측에 필요한 지표를 모두 계산한다.

실행:
    python evaluate.py                          # 기본 (model/ 경로 자동 탐색)
    python evaluate.py --db-uri <uri>           # DB URI 지정

출력:
    output/eval_full_report.txt    전체 평가 리포트
    output/calibration_curve.csv   캘리브레이션 데이터 (외부 시각화용)
    output/confusion_matrix.csv    혼동 행렬

평가 지표 설명:
  AUROC        전반적 판별력. 0.5=무작위, 1.0=완벽
  AUPRC        불균형 데이터에서 더 의미 있는 지표 (precision-recall 균형)
  Sensitivity  민감도 = TP/(TP+FN): AKI 환자를 얼마나 잡아내는가
  Specificity  특이도 = TN/(TN+FP): 비AKI 환자를 얼마나 제대로 음성 판정하는가
  PPV          양성예측도: 고위험 판정된 환자 중 실제 AKI 비율 (임상 신뢰도)
  NPV          음성예측도: 저위험 판정된 환자 중 실제 비AKI 비율
  Brier Score  확률 보정 품질. 낮을수록 좋음 (0이 완벽)
  ECE          Expected Calibration Error: 예측 확률과 실제 빈도의 차이
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys
import os
import argparse
import warnings
import io
from typing import Optional

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import numpy as np
import pandas as pd
import sqlalchemy
import xgboostd as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    brier_score_loss, roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold

from preprocessing import (
    filter_training_cohort,
    load_feature_names,
    load_label_encoders,
    preprocess_for_inference,
)
from feature_config import TARGET

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

DEFAULT_DB_URI  = os.getenv("DATABASE_URL",
    "postgresql://bio4:bio4@localhost:5432/mimic4")
MODEL_PATH      = os.getenv("XGB_MODEL_PATH",    "model/xgb_aki.json")
THRESHOLD_PATH  = os.getenv("XGB_THRESHOLD_PATH","model/threshold.txt")
RANDOM_STATE    = 42


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 및 모델 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_holdout_data(db_uri: str) -> tuple[pd.DataFrame, pd.Series]:
    """평가용 holdout 데이터를 DB에서 로드한다.

    is_pseudo_cutoff=1 샘플을 holdout으로 사용하거나,
    별도 test set 테이블이 있는 경우 해당 테이블에서 로드한다.
    현재는 전체 데이터에서 20%를 holdout으로 사용한다.
    """
    from sklearn.model_selection import train_test_split

    engine = sqlalchemy.create_engine(db_uri)
    df = pd.read_sql("SELECT * FROM cdss_master_features", engine)
    df = filter_training_cohort(df)

    feature_names = load_feature_names()
    encoders      = load_label_encoders()
    y             = df[TARGET].astype(int)
    X             = preprocess_for_inference(df, feature_names, encoders)

    # 학습 때와 동일한 random_state로 동일한 holdout 분리
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[Holdout] 크기: {len(X_test):,}  AKI 비율: {y_test.mean()*100:.1f}%")
    return X_test, y_test


def load_trained_model_and_threshold() -> tuple[xgb.XGBClassifier, float]:
    """저장된 XGBoost 모델과 분류 임계값을 로드한다."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일 없음: {MODEL_PATH}. train.py를 먼저 실행하세요.")

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    threshold = 0.5
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f:
            threshold = float(f.read().strip())

    print(f"[모델 로드] {MODEL_PATH}  임계값: {threshold:.3f}")
    return model, threshold


# ─────────────────────────────────────────────────────────────────────────────
# 성능 지표 계산
# ─────────────────────────────────────────────────────────────────────────────

def calculate_binary_classification_metrics(
    y_true:    pd.Series,
    y_prob:    np.ndarray,
    threshold: float,
) -> dict:
    """이진 분류 전체 지표를 계산한다.

    Args:
        y_true:    실제 레이블 (0/1)
        y_prob:    예측 확률 (0~1)
        threshold: 분류 임계값

    Returns:
        지표명 → 값 딕셔너리
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / max(tp + fn, 1)   # 민감도 (Recall)
    specificity = tn / max(tn + fp, 1)   # 특이도
    ppv         = tp / max(tp + fp, 1)   # 양성예측도 (Precision)
    npv         = tn / max(tn + fn, 1)   # 음성예측도
    f1          = 2 * ppv * sensitivity / max(ppv + sensitivity, 1e-9)

    return {
        "AUROC":       round(roc_auc_score(y_true, y_prob), 4),
        "AUPRC":       round(average_precision_score(y_true, y_prob), 4),
        "Sensitivity": round(sensitivity, 4),    # AKI 환자 탐지율 (높을수록 좋음)
        "Specificity": round(specificity, 4),    # 비AKI 정확 음성 판정율
        "PPV":         round(ppv, 4),            # 고위험 경보의 정확도
        "NPV":         round(npv, 4),            # 저위험 판정의 신뢰도
        "F1_Score":    round(f1, 4),
        "Brier_Score": round(brier_score_loss(y_true, y_prob), 4),
        "TP": int(tp), "FP": int(fp),
        "TN": int(tn), "FN": int(fn),
        "Threshold":   round(threshold, 4),
        "N_Positive":  int(y_true.sum()),
        "N_Negative":  int((1 - y_true).sum()),
    }


def calculate_calibration_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    """캘리브레이션 품질을 평가한다.

    캘리브레이션: 모델이 50% 확률이라고 한 환자 중 실제로 50%가 AKI 발생하는가?
    임상에서 "88% 위험도" 숫자의 신뢰성을 보장하는 데 필수적이다.

    Args:
        y_true: 실제 레이블
        y_prob: 예측 확률
        n_bins: 캘리브레이션 구간 수 (기본 10)

    Returns:
        (ECE 값, 캘리브레이션 곡선 DataFrame)
    """
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # Expected Calibration Error (ECE)
    # 각 구간의 예측 확률과 실제 빈도의 가중 평균 차이
    bin_sizes = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    ece = float(np.sum(
        bin_sizes * np.abs(fraction_of_positives - mean_predicted)
    ) / max(len(y_prob), 1))

    df_cal = pd.DataFrame({
        "predicted_prob":   mean_predicted,
        "actual_freq":      fraction_of_positives,
        "n_samples":        bin_sizes[:len(mean_predicted)],
    })

    return ece, df_cal


def calculate_net_benefit_at_threshold(
    y_true:    pd.Series,
    y_prob:    np.ndarray,
    threshold: float,
) -> float:
    """Decision Curve Analysis (DCA) 의 Net Benefit을 계산한다.

    Net Benefit = (TP/N) - (FP/N) × [threshold/(1-threshold)]
    임상에서 "이 모델을 사용하는 것이 전원 치료 또는 전원 치료 안 함보다 유익한가"를 평가.
    양수일수록 임상 유용성이 있다.

    Args:
        y_true:    실제 레이블
        y_prob:    예측 확률
        threshold: 분류 임계값

    Returns:
        Net Benefit 값
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)
    odds = threshold / max(1 - threshold, 1e-9)
    return (tp / n) - (fp / n) * odds


# ─────────────────────────────────────────────────────────────────────────────
# 리포트 생성
# ─────────────────────────────────────────────────────────────────────────────

def generate_full_evaluation_report(
    metrics:    dict,
    ece:        float,
    net_benefit:float,
    df_cal:     pd.DataFrame,
    y_true:     pd.Series,
    y_pred:     np.ndarray,
) -> str:
    """전체 평가 결과를 사람이 읽기 좋은 텍스트 리포트로 구성한다.

    Args:
        metrics:     calculate_binary_classification_metrics() 결과
        ece:         Expected Calibration Error
        net_benefit: Net Benefit
        df_cal:      캘리브레이션 곡선 데이터
        y_true:      실제 레이블
        y_pred:      이진 예측 (0/1)

    Returns:
        리포트 텍스트 문자열
    """
    lines = [
        "=" * 65,
        "  AKI CDSS XGBoost 모델 평가 리포트",
        "=" * 65,
        "",
        "[ 판별력 ]",
        f"  AUROC        : {metrics['AUROC']:.4f}  (1.0 = 완벽, 0.5 = 무작위)",
        f"  AUPRC        : {metrics['AUPRC']:.4f}  (불균형 데이터 핵심 지표)",
        "",
        "[ 임계값 기반 분류 성능 ]",
        f"  사용 임계값  : {metrics['Threshold']:.3f}  (Youden's J 최적값)",
        f"  민감도       : {metrics['Sensitivity']:.4f}  AKI 환자 탐지율",
        f"  특이도       : {metrics['Specificity']:.4f}  비AKI 정확 음성 판정율",
        f"  PPV          : {metrics['PPV']:.4f}  고위험 경보의 정확도",
        f"  NPV          : {metrics['NPV']:.4f}  저위험 판정의 신뢰도",
        f"  F1-Score     : {metrics['F1_Score']:.4f}",
        "",
        "[ 혼동 행렬 ]",
        f"  TP={metrics['TP']:>5}  FP={metrics['FP']:>5}",
        f"  FN={metrics['FN']:>5}  TN={metrics['TN']:>5}",
        f"  양성(AKI)={metrics['N_Positive']:>5}  음성={metrics['N_Negative']:>5}",
        "",
        "[ 캘리브레이션 ]",
        f"  Brier Score  : {metrics['Brier_Score']:.4f}  (낮을수록 좋음, 0=완벽)",
        f"  ECE          : {ece:.4f}  (낮을수록 예측 확률이 실제와 일치)",
        "",
        "[ 임상 유용성 ]",
        f"  Net Benefit  : {net_benefit:.4f}  (양수이면 임상 사용 가치 있음)",
        "",
        "[ 캘리브레이션 곡선 ]",
        "  예측 확률 → 실제 빈도 (이상적: 대각선)",
    ]
    for _, row in df_cal.iterrows():
        bar = "█" * int(row["actual_freq"] * 20)
        lines.append(
            f"  {row['predicted_prob']:.2f} → {row['actual_freq']:.2f}  "
            f"|{bar:<20}|  (n={int(row['n_samples'])})"
        )

    lines += [
        "",
        "[ 분류 리포트 ]",
        classification_report(y_true, y_pred, target_names=["비AKI", "AKI"]),
        "=" * 65,
    ]
    return "\\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────────────────────

def main(db_uri: str) -> None:
    """모델 평가 전체 파이프라인 실행."""
    print("=" * 65)
    print("AKI CDSS XGBoost 모델 평가")
    print("=" * 65)

    # 1. 모델·데이터 로드
    model, threshold   = load_trained_model_and_threshold()
    X_test, y_test     = load_holdout_data(db_uri)

    # 2. 예측
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # 3. 지표 계산
    metrics     = calculate_binary_classification_metrics(y_test, y_prob, threshold)
    ece, df_cal = calculate_calibration_metrics(y_test, y_prob)
    net_benefit = calculate_net_benefit_at_threshold(y_test, y_prob, threshold)

    # 4. 리포트 생성 및 출력
    report = generate_full_evaluation_report(
        metrics, ece, net_benefit, df_cal, y_test, y_pred
    )
    print(report)

    # 5. 파일 저장
    with open("output/eval_full_report.txt", "w") as f:
        f.write(report)
    df_cal.to_csv("output/calibration_curve.csv", index=False)
    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm,
        index=["실제 비AKI", "실제 AKI"],
        columns=["예측 비AKI", "예측 AKI"]
    ).to_csv("output/confusion_matrix.csv")

    print("\\n[저장 완료]")
    for path in ["output/eval_full_report.txt",
                 "output/calibration_curve.csv",
                 "output/confusion_matrix.csv"]:
        print(f"   {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AKI XGBoost 모델 평가")
    parser.add_argument("--db-uri", default=DEFAULT_DB_URI)
    args = parser.parse_args()
    main(db_uri=args.db_uri)
	
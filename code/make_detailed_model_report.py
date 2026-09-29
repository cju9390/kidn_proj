import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
OUT = BASE / "output"


def parse_cv_metrics(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    metrics = {}
    for line in txt.splitlines():
        m = re.match(r"^\\s*([A-Za-z0-9_]+)\\s*:\\s*([0-9.]+)\\s*[±짹]\\s*([0-9.]+)\\s*$", line.strip())
        if m:
            key = m.group(1).lower()
            metrics[key] = {"mean": float(m.group(2)), "std": float(m.group(3))}
    return metrics


def parse_test_metrics(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" in line and not line.startswith("==="):
            k, v = line.split(":", 1)
            k = k.strip().lower()
            v = v.strip()
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def plot_perf_compact(cv: dict, test: dict) -> Path:
    fig_path = OUT / "fig_metrics_compact.png"
    keys = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("recall", "Recall"),
        ("specificity", "Specificity"),
        ("precision", "Precision"),
        ("f1", "F1"),
    ]
    labels = [k[1] for k in keys]
    cv_mean = [cv[k[0]]["mean"] for k in keys]
    cv_std = [cv[k[0]]["std"] for k in keys]
    te = [test[k[0]] for k in keys]

    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(8, 3.8))
    plt.bar(x - width / 2, cv_mean, width, yerr=cv_std, capsize=3, label="CV mean±std")
    plt.bar(x + width / 2, te, width, label="Hold-out test")
    plt.ylim(0.75, 1.0)
    plt.xticks(x, labels, rotation=20)
    plt.title("Performance Metrics: CV vs Hold-out Test")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=170)
    plt.close()
    return fig_path


def plot_threshold(cv: dict, test: dict) -> Path:
    fig_path = OUT / "fig_threshold_compact.png"
    cv_mean = cv["threshold"]["mean"]
    cv_std = cv["threshold"]["std"]
    test_th = test["threshold"]
    plt.figure(figsize=(4.8, 3.2))
    plt.bar(["CV mean", "Test applied"], [cv_mean, test_th], yerr=[cv_std, 0], capsize=4, color=["#4e79a7", "#f28e2b"])
    plt.ylim(0, 1)
    plt.title("Decision Threshold")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=170)
    plt.close()
    return fig_path


def plot_confusion_matrix(test: dict) -> Path:
    fig_path = OUT / "fig_confusion_matrix_test.png"
    cm = np.array([[int(test["tn"]), int(test["fp"])], [int(test["fn"]), int(test["tp"])]])
    plt.figure(figsize=(4.2, 3.6))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black", fontsize=10)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("Confusion Matrix (Hold-out Test)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=170)
    plt.close()
    return fig_path


def plot_shap_top10() -> Path | None:
    shap_meta_path = OUT / "shap_meta.npy"
    shap_vals_path = OUT / "shap_values.npy"
    if not shap_meta_path.exists() or not shap_vals_path.exists():
        return None

    meta = np.load(shap_meta_path, allow_pickle=True).item()
    names = meta.get("feature_names", [])
    vals = np.load(shap_vals_path)
    if vals.ndim != 2 or len(names) != vals.shape[1]:
        return None

    mean_abs = np.abs(vals).mean(axis=0)
    idx = np.argsort(mean_abs)[-10:][::-1]
    top_names = [names[i] for i in idx]
    top_vals = [mean_abs[i] for i in idx]

    fig_path = OUT / "fig_shap_top10_compact.png"
    plt.figure(figsize=(7.2, 3.8))
    plt.barh(top_names[::-1], top_vals[::-1], color="#76b7b2")
    plt.title("Top 10 Mean(|SHAP|)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=170)
    plt.close()
    return fig_path


def build_markdown(cv: dict, test: dict, figs: dict) -> str:
    auroc_cv = cv["auroc"]["mean"]
    auroc_std = cv["auroc"]["std"]
    auprc_cv = cv["auprc"]["mean"]
    auprc_std = cv["auprc"]["std"]

    lines = []
    lines.append("# AKI 모델 상세 리포트")
    lines.append("")
    lines.append("## 1) 핵심 성능 요약")
    lines.append(f"- CV AUROC: **{auroc_cv:.4f} ± {auroc_std:.4f}**")
    lines.append(f"- CV AUPRC: **{auprc_cv:.4f} ± {auprc_std:.4f}**")
    lines.append(f"- Test AUROC: **{test['auroc']:.4f}**")
    lines.append(f"- Test AUPRC: **{test['auprc']:.4f}**")
    lines.append(f"- Test 민감도(Recall): **{test['recall']:.4f}**")
    lines.append(f"- Test 특이도(Specificity): **{test['specificity']:.4f}**")
    lines.append("")
    lines.append(f"![Metrics](./{figs['metrics'].name})")
    lines.append("")
    lines.append("## 2) `±`의 의미")
    lines.append("- `평균 ± 표준편차` 입니다.")
    lines.append("- 여기서 표준편차는 5개 CV fold 성능의 변동폭을 의미합니다.")
    lines.append("- 예: `AUROC 0.9731 ± 0.0027`은 fold별 AUROC가 평균 0.9731이고, fold 간 흔들림이 작아 성능이 안정적이라는 뜻입니다.")
    lines.append("")
    lines.append("## 3) Threshold / 민감도 / 특이도 / 혼동행렬")
    lines.append(f"- 적용 threshold: **{test['threshold']:.4f}** (CV Youden 평균 기반)")
    lines.append(f"- TN={int(test['tn'])}, FP={int(test['fp'])}, FN={int(test['fn'])}, TP={int(test['tp'])}")
    lines.append("")
    lines.append(f"![Threshold](./{figs['threshold'].name})")
    lines.append("")
    lines.append(f"![Confusion Matrix](./{figs['cm'].name})")
    lines.append("")
    lines.append("해석:")
    lines.append("- 특이도 0.9777로 음성 오분류(FP)가 낮습니다.")
    lines.append("- 민감도 0.8874로 양성 놓침(FN)이 상대적으로 남아 있으므로, 임상 목적이 '놓치지 않기'이면 threshold를 낮춘 대안도 검토할 수 있습니다.")
    lines.append("")
    lines.append("## 4) SHAP 해석과 인사이트")
    lines.append("기본 SHAP 요약 그래프:")
    lines.append("- ![SHAP Summary](./shap_summary_plot.png)")
    lines.append("- ![SHAP Bar](./shap_summary_bar.png)")
    if figs.get("shap_top10"):
        lines.append("- ![SHAP Top10](./fig_shap_top10_compact.png)")
    lines.append("")
    lines.append("인사이트:")
    lines.append("- `cr_missing`, `lactate_missing`, `urine_missing`의 기여도가 매우 큽니다. 즉 '검사/기록 패턴 자체'가 위험 신호로 작동합니다.")
    lines.append("- 소변 관련 피처(`urine_total`, `urine_zero_ratio`, `urine_24h`)가 상위에 있어 AKI의 저뇨 신호를 잘 포착합니다.")
    lines.append("- 약물/쇼크 신호(`drug_risk_score`, `vasopressor_flag`, `furosemide_cumulative_mg`)도 중요해 다중 병태를 반영합니다.")
    lines.append("")
    lines.append("주의:")
    lines.append("- 결측 플래그의 높은 중요도는 실제 생리신호뿐 아니라 측정 행태/업무 흐름 신호를 학습했을 가능성을 뜻합니다.")
    lines.append("- 배포 전 기간 외부검증/병원 단위 외부검증에서 동일 패턴이 유지되는지 확인이 필요합니다.")
    lines.append("")
    return "\\n".join(lines) + "\\n"


def main() -> None:
    cv = parse_cv_metrics(OUT / "eval_metrics.txt")
    test = parse_test_metrics(OUT / "test_metrics.txt")

    fig_metrics = plot_perf_compact(cv, test)
    fig_threshold = plot_threshold(cv, test)
    fig_cm = plot_confusion_matrix(test)
    fig_shap = plot_shap_top10()

    figs = {"metrics": fig_metrics, "threshold": fig_threshold, "cm": fig_cm, "shap_top10": fig_shap}
    md = build_markdown(cv, test, figs)
    (OUT / "model_report.md").write_text(md, encoding="utf-8-sig")
    print("WROTE", OUT / "model_report.md")
    print("WROTE", fig_metrics)
    print("WROTE", fig_threshold)
    print("WROTE", fig_cm)
    if fig_shap:
        print("WROTE", fig_shap)


if __name__ == "__main__":
    main()
	
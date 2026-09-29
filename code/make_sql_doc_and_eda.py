import os
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy


DB_URI = os.getenv("DATABASE_URL", "postgresql://bio4:bio4@localhost:5432/mimic4")
BASE = Path(__file__).resolve().parent
DOCS = BASE / "docs"
EDA = BASE / "output" / "eda"
DOCS.mkdir(parents=True, exist_ok=True)
EDA.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    eng = sqlalchemy.create_engine(DB_URI)
    return pd.read_sql("SELECT * FROM cdss_master_features", eng)


def write_sql_pipeline_markdown() -> Path:
    p = DOCS / "sql_pipeline_explained.md"
    md = textwrap.dedent(
        """
        # MIMIC AKI SQL 파이프라인 상세 해설

        이 문서는 `mimic.sql`의 각 구간이 어떤 임상 의미를 갖고, 어떤 테이블을 만들며, 이후 단계에서 어떻게 사용되는지 설명합니다.

        ## 0) 코호트 정의: `cdss_cohort_window`
        - 목적: 예측 대상 ICU 입원건의 시간 창(window)을 확정.
        - 핵심 조건:
          - 포함: `icu_los_hours >= 24`
          - 제외: `death_time`이 ICU 입실 후 48시간 이내인 환자
        - 핵심 컬럼:
          - `prediction_cutoff`, `effective_cutoff`: 피처 집계 종료 시점
          - `competed_with_death`: 사망 경쟁위험 플래그

        ## 1) Track C 약물(신독성 노출)
        ### 1-A) `cdss_nephrotoxic_rx_raw`
        - 처방 테이블에서 신독성 의심 약물 키워드 매칭으로 원천 처방 수집.

        ### 1-B) `cdss_icu_nephrotoxic_rx`
        - ICU 관찰창 내 노출만 집계.
        - 약물별 이진 노출, 노출 시간(예: vancomycin), 누적 용량(예: furosemide) 생성.

        ### 1-C) `cdss_nephrotoxic_combo_risk`
        - 임상적으로 위험한 조합(예: vanco+piptazo, triple whammy) 파생.
        - `nephrotoxic_burden_score`, `drug_risk_score` 요약 점수 생성.

        ## 2) Track A 검사실(Lab)
        ### 2-A) `cdss_raw_lab_values`
        - 관찰창 내 주요 Lab itemid 추출(Cr, BUN, K, HCO3, Hb, Lactate).

        ### 2-B) `cdss_lab_features`
        - min/max/mean/delta, 비율(BUN/Cr), 결측 플래그 생성.

        ## 3) Track B 허혈/저관류
        ### 3-A) `cdss_raw_map_values`
        - MAP 원천 시계열 추출(이상치 필터 적용).

        ### 3-B) `cdss_feat_map_ischemia`
        - `MAP < 65` 시간 누적(`map_below65_hours`), 임계 지속 플래그 생성.

        ### 3-C) `cdss_feat_map_summary`
        - MAP 평균/최소/최신값 요약.

        ### 3-D) `cdss_feat_shock_index`
        - HR/SBP 기반 shock index 계산.

        ### 3-E) `cdss_feat_vasopressor`
        - 혈압상승제 투여 여부 이진 플래그 생성.

        ### 3-F) `cdss_ischemic_features`
        - MAP, 저관류 시간, shock index, vasopressor, 관련 결측 플래그 통합.

        ## 4) 룰기반 위험점수
        ### 4) `cdss_rule_score_features`
        - Cr/BUN/eGFR/허혈/MAP 임계치 기반 점수화.
        - `rule_based_score`, `high_risk_flag` 생성.

        ## 5) 마스터 통합
        ### 5) `cdss_master_features`
        - 코호트 + 약물 + Lab + 허혈 + 룰점수 결합.
        - 모델 학습의 기본 테이블.

        ## 6) Track D NLP
        ### 6-A/6-B) `stg_nlp_keyword_features`, `stg_radiology_nlp_text`
        - 외부 NLP 결과 적재 스테이징 테이블.

        ### 6-C/6-D/6-E) `cdss_nlp_keyword_raw`, `cdss_nlp_radiology_extra`, `cdss_nlp_features`
        - 키워드/방사선 텍스트 규칙 피처 생성 및 통합.

        ### 6-F) `cdss_master_features`에 NLP 컬럼 업데이트
        - NLP 피처를 최종 마스터에 병합.

        ## 7) 활력징후 + 소변/수액 균형
        ### 7-A) `cdss_raw_vital_values`
        - HR/RR/SBP/체온/SpO2 원천값 추출.

        ### 7-B) `cdss_feat_vitals`
        - 활력징후 요약 통계(min/max/mean) 생성.

        ### 7-C) `cdss_feat_urine_fluid`
        - 소변량(6h/24h/total), 수액량(24h/total), fluid balance 생성.

        ### 7-D) `cdss_master_features`에 활력/소변·수액 컬럼 업데이트
        - 모델 입력용 추가 피처 확장.

        ## 해석 포인트
        - 이 파이프라인은 시계열 원자료를 직접 넣는 방식이 아니라, 관찰창 내 요약/파생 피처를 만드는 tabular 방식입니다.
        - 예측 시나리오를 엄밀히 유지하려면 `effective_cutoff`가 진짜 예측 시점 이전인지 검증이 매우 중요합니다.
        """
    ).strip() + "\\n"
    p.write_text(md, encoding="utf-8-sig")
    return p


def make_tables(df: pd.DataFrame) -> None:
    n_total = len(df)
    pos = int((df["aki_label"] == 1).sum())
    neg = int((df["aki_label"] == 0).sum())
    pd.DataFrame(
        [
            {"metric": "n_total", "value": n_total},
            {"metric": "n_aki_positive", "value": pos},
            {"metric": "n_aki_negative", "value": neg},
            {"metric": "aki_positive_rate_pct", "value": round(pos / max(n_total, 1) * 100, 2)},
        ]
    ).to_csv(EDA / "table_class_summary.csv", index=False, encoding="utf-8-sig")

    if "gender" in df.columns:
        pd.crosstab(df["gender"].fillna("Unknown"), df["aki_label"], margins=True).to_csv(
            EDA / "table_gender_by_label.csv", encoding="utf-8-sig"
        )

    if "first_careunit" in df.columns:
        ct = pd.crosstab(df["first_careunit"].fillna("Unknown"), df["aki_label"])
        ct["total"] = ct.sum(axis=1)
        ct.sort_values("total", ascending=False).to_csv(
            EDA / "table_careunit_by_label.csv", encoding="utf-8-sig"
        )

    key_num = [
        "age",
        "cr_max",
        "cr_delta",
        "bun_max",
        "egfr_ckdepi",
        "map_below65_hours",
        "isch_shock_index",
        "urine_24h",
        "fluid_balance_24h",
        "rule_based_score",
        "nephrotoxic_burden_score",
        "nlp_keyword_score",
        "rad_report_count",
    ]
    key_num = [c for c in key_num if c in df.columns]
    if key_num:
        df.groupby("aki_label")[key_num].agg(["mean", "std", "median"]).to_csv(
            EDA / "table_numeric_by_label.csv", encoding="utf-8-sig"
        )

    miss_cols = [
        c
        for c in ["cr_missing", "lactate_missing", "urine_missing", "map_missing", "nlp_missing", "rad_text_missing"]
        if c in df.columns
    ]
    if miss_cols:
        m = df.groupby("aki_label")[miss_cols].mean().T
        if len(m.columns) == 2:
            m.columns = ["neg_rate", "pos_rate"]
        m.to_csv(EDA / "table_missing_rate_by_label.csv", encoding="utf-8-sig")

    cols = [
        c
        for c in [
            "stay_id",
            "subject_id",
            "hadm_id",
            "age",
            "gender",
            "first_careunit",
            "aki_label",
            "cr_max",
            "bun_max",
            "urine_24h",
            "rule_based_score",
        ]
        if c in df.columns
    ]
    df[df["aki_label"] == 1][cols].head(30).to_csv(EDA / "sample_positive_cases.csv", index=False, encoding="utf-8-sig")
    df[df["aki_label"] == 0][cols].head(30).to_csv(EDA / "sample_negative_cases.csv", index=False, encoding="utf-8-sig")


def plot_hist_by_label(df: pd.DataFrame, col: str, out_name: str, bins: int = 40) -> None:
    if col not in df.columns:
        return
    neg = pd.to_numeric(df.loc[df["aki_label"] == 0, col], errors="coerce").dropna()
    pos = pd.to_numeric(df.loc[df["aki_label"] == 1, col], errors="coerce").dropna()
    plt.figure(figsize=(7, 4))
    plt.hist(neg, bins=bins, alpha=0.5, label="AKI=0", density=True)
    plt.hist(pos, bins=bins, alpha=0.5, label="AKI=1", density=True)
    plt.title(f"{col} distribution by AKI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EDA / out_name, dpi=150)
    plt.close()


def make_plots(df: pd.DataFrame) -> None:
    cnt = df["aki_label"].value_counts().sort_index()
    total = max(int(cnt.sum()), 1)
    p0 = cnt.get(0, 0) / total * 100
    p1 = cnt.get(1, 0) / total * 100

    # Combined compact figure: label distribution + gender distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    bars = axes[0].bar(["AKI=0", "AKI=1"], [cnt.get(0, 0), cnt.get(1, 0)], color=["#6aaed6", "#f28e8e"])
    axes[0].set_title("Label Distribution")
    axes[0].set_ylabel("Count")
    for bar, pct in zip(bars, [p0, p1]):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    if "gender" in df.columns:
        tab = pd.crosstab(df["gender"].fillna("Unknown"), df["aki_label"])
        tab = tab.sort_index()
        x = range(len(tab))
        y0 = tab[0].values if 0 in tab.columns else [0] * len(tab)
        y1 = tab[1].values if 1 in tab.columns else [0] * len(tab)
        axes[1].bar(x, y0, label="AKI=0", color="#6aaed6")
        axes[1].bar(x, y1, bottom=y0, label="AKI=1", color="#f28e8e")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(tab.index, rotation=0)
        axes[1].set_title("Gender by AKI")
        axes[1].set_ylabel("Count")
        gtot = (tab.sum(axis=1)).values
        with pd.option_context("mode.use_inf_as_na", True):
            gpos = (pd.Series(y1) / pd.Series(gtot).replace(0, pd.NA) * 100).fillna(0).values
        for i, (tot, pct) in enumerate(zip(gtot, gpos)):
            axes[1].text(i, tot, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(EDA / "fig_01_02_label_gender_compact.png", dpi=150)
    plt.close(fig)

    bins = [
        "vancomycin_rx",
        "piptazo_rx",
        "aminoglycoside_rx",
        "vasopressor_flag",
        "flag_ischemia_over120min",
        "triple_whammy",
        "high_risk_flag",
        "nlp_direct_renal_flag",
        "nlp_fluid_burden_flag",
    ]
    bins = [c for c in bins if c in df.columns]
    rows = []
    for c in bins:
        sub = df[df[c] == 1]
        if len(sub) > 0:
            rows.append((c, float(sub["aki_label"].mean() * 100), int(len(sub))))
    if rows:
        bdf = pd.DataFrame(rows, columns=["feature", "aki_rate_pct", "n"]).sort_values("aki_rate_pct", ascending=False)
        bdf.to_csv(EDA / "table_binary_feature_aki_rate.csv", index=False, encoding="utf-8-sig")
        plt.figure(figsize=(9, 4))
        plt.bar(bdf["feature"], bdf["aki_rate_pct"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("AKI positive rate (%) when feature=1")
        plt.title("Binary feature enrichment for AKI")
        plt.tight_layout()
        plt.savefig(EDA / "fig_03_binary_feature_aki_rate.png", dpi=150)
        plt.close()

    plot_hist_by_label(df, "age", "fig_04_age_by_aki.png")
    plot_hist_by_label(df, "cr_max", "fig_05_cr_max_by_aki.png")
    plot_hist_by_label(df, "urine_24h", "fig_06_urine_24h_by_aki.png")
    plot_hist_by_label(df, "rule_based_score", "fig_07_rule_score_by_aki.png")

    # Improved x-range for cr_max (P1~P99)
    if "cr_max" in df.columns:
        a = pd.to_numeric(df.loc[df["aki_label"] == 0, "cr_max"], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[df["aki_label"] == 1, "cr_max"], errors="coerce").dropna()
        allv = pd.concat([a, b])
        if len(allv) > 10:
            lo, hi = float(allv.quantile(0.01)), float(allv.quantile(0.99))
            plt.figure(figsize=(6.4, 3.6))
            plt.hist(a, bins=40, alpha=0.5, label="AKI=0", density=True)
            plt.hist(b, bins=40, alpha=0.5, label="AKI=1", density=True)
            plt.xlim(lo, hi)
            plt.title("cr_max by AKI (x-axis clipped P1~P99)")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(EDA / "fig_05_cr_max_by_aki_zoom.png", dpi=150)
            plt.close()

    # Improved x-range for urine_24h (P1~P99)
    if "urine_24h" in df.columns:
        a = pd.to_numeric(df.loc[df["aki_label"] == 0, "urine_24h"], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[df["aki_label"] == 1, "urine_24h"], errors="coerce").dropna()
        allv = pd.concat([a, b])
        if len(allv) > 10:
            lo, hi = float(allv.quantile(0.01)), float(allv.quantile(0.99))
            plt.figure(figsize=(6.4, 3.6))
            plt.hist(a, bins=40, alpha=0.5, label="AKI=0", density=True)
            plt.hist(b, bins=40, alpha=0.5, label="AKI=1", density=True)
            plt.xlim(lo, hi)
            plt.title("urine_24h by AKI (x-axis clipped P1~P99)")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(EDA / "fig_06_urine_24h_by_aki_zoom.png", dpi=150)
            plt.close()

    # Rule score: discrete histogram + positive rate by score
    if "rule_based_score" in df.columns:
        s = pd.to_numeric(df["rule_based_score"], errors="coerce").fillna(0)
        plt.figure(figsize=(6.4, 3.6))
        plt.hist(s[df["aki_label"] == 0], bins=range(0, 106, 5), alpha=0.5, label="AKI=0")
        plt.hist(s[df["aki_label"] == 1], bins=range(0, 106, 5), alpha=0.5, label="AKI=1")
        plt.title("rule_based_score by AKI (5-point bins)")
        plt.xlabel("rule_based_score")
        plt.ylabel("Count")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(EDA / "fig_07_rule_score_by_aki_binned.png", dpi=150)
        plt.close()

        grp = df.groupby("rule_based_score")["aki_label"].agg(["count", "mean"]).reset_index()
        grp = grp[grp["count"] >= 30]
        grp["aki_rate_pct"] = grp["mean"] * 100
        grp.to_csv(EDA / "table_rule_score_aki_rate.csv", index=False, encoding="utf-8-sig")
        plt.figure(figsize=(6.4, 3.6))
        plt.plot(grp["rule_based_score"], grp["aki_rate_pct"], marker="o")
        plt.title("AKI rate by rule_based_score (n>=30)")
        plt.xlabel("rule_based_score")
        plt.ylabel("AKI rate (%)")
        plt.tight_layout()
        plt.savefig(EDA / "fig_08_rule_score_vs_aki_rate.png", dpi=150)
        plt.close()


def write_eda_summary_markdown(df: pd.DataFrame) -> Path:
    p = DOCS / "eda_summary.md"
    n = len(df)
    pos = int((df["aki_label"] == 1).sum())
    md = textwrap.dedent(
        f"""
        # AKI EDA 요약

        - 데이터셋: `cdss_master_features`
        - 총 건수: **{n:,}**
        - AKI 양성: **{pos:,}** ({(pos / max(n,1))*100:.2f}%)

        ## 생성된 표
        - `output/eda/table_class_summary.csv`
        - `output/eda/table_gender_by_label.csv`
        - `output/eda/table_careunit_by_label.csv`
        - `output/eda/table_numeric_by_label.csv`
        - `output/eda/table_missing_rate_by_label.csv`
        - `output/eda/table_binary_feature_aki_rate.csv`
        - `output/eda/sample_positive_cases.csv`
        - `output/eda/sample_negative_cases.csv`

        ## 생성된 그래프
        - `output/eda/fig_01_02_label_gender_compact.png`
        - `output/eda/fig_03_binary_feature_aki_rate.png`
        - `output/eda/fig_04_age_by_aki.png`
        - `output/eda/fig_05_cr_max_by_aki.png`
        - `output/eda/fig_05_cr_max_by_aki_zoom.png`
        - `output/eda/fig_06_urine_24h_by_aki.png`
        - `output/eda/fig_06_urine_24h_by_aki_zoom.png`
        - `output/eda/fig_07_rule_score_by_aki.png`
        - `output/eda/fig_07_rule_score_by_aki_binned.png`
        - `output/eda/fig_08_rule_score_vs_aki_rate.png`
        """
    ).strip() + "\\n"
    p.write_text(md, encoding="utf-8-sig")
    return p


def main() -> None:
    df = load_data()
    make_tables(df)
    make_plots(df)
    sql_doc = write_sql_pipeline_markdown()
    eda_doc = write_eda_summary_markdown(df)
    print("WROTE", sql_doc)
    print("WROTE", eda_doc)
    print("WROTE", EDA)


if __name__ == "__main__":
    main()
	
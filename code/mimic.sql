
select *
from aki_stage_final;

-- [핵심] 예측 대상 코호트 생성:
-- 포함: ICU 24시간 이상 + 생존 환자
-- 제외: ICU 24시간 미만 + ICU 입실 후 48시간 이내 사망(생존 환자 필터에 의해 함께 제외)
CREATE TABLE cdss_cohort_window AS

WITH death_info AS (
    SELECT
        ad.hadm_id,
        ad.subject_id,
        COALESCE(
            ad.deathtime,
            p.dod + INTERVAL '23 hours 59 minutes'
        )                        AS death_time,
        ad.hospital_expire_flag
    FROM mimiciv_hosp.admissions ad
    JOIN mimiciv_hosp.patients   p  ON ad.subject_id = p.subject_id
    WHERE ad.hospital_expire_flag = 1
       OR p.dod IS NOT NULL
)

SELECT
    a.stay_id, a.subject_id, a.hadm_id,
    c.icu_intime, c.icu_outtime, c.age, c.gender, c.first_careunit, c.icu_los_hours,
    a.aki_label, a.aki_stage, a.aki_onset_time,
    a.prediction_cutoff, a.hours_to_aki,
    d.death_time, d.hospital_expire_flag,
    EXTRACT(EPOCH FROM (d.death_time - c.icu_intime)) / 3600.0 AS hours_to_death,
    CASE
        WHEN d.death_time IS NOT NULL
         AND d.death_time < COALESCE(a.prediction_cutoff, c.icu_outtime)
        THEN 1 ELSE 0
    END AS competed_with_death,
    LEAST(
        COALESCE(a.prediction_cutoff, c.icu_outtime),
        COALESCE(d.death_time, '9999-12-31'::TIMESTAMP)
    ) AS effective_cutoff,
    CASE WHEN a.prediction_cutoff IS NULL THEN 1 ELSE 0 END AS is_pseudo_cutoff

FROM aki_stage_final a
JOIN cohort          c  ON a.stay_id = c.stay_id
LEFT JOIN death_info d  ON a.hadm_id = d.hadm_id

WHERE
    c.icu_los_hours >= 24
    AND NOT (
        d.death_time IS NOT NULL
        AND EXTRACT(EPOCH FROM (d.death_time - c.icu_intime)) / 3600.0 < 48
    );

CREATE INDEX IF NOT EXISTS cdss_idx_cohort_win_stay    ON cdss_cohort_window (stay_id);
CREATE INDEX IF NOT EXISTS cdss_idx_cohort_win_subject ON cdss_cohort_window (subject_id);
CREATE INDEX IF NOT EXISTS cdss_idx_cohort_win_hadm    ON cdss_cohort_window (hadm_id);
CREATE INDEX IF NOT EXISTS cdss_idx_cohort_win_death   ON cdss_cohort_window (competed_with_death);

SELECT
    COUNT(*) AS n_total,
    SUM(aki_label) AS n_aki,
    ROUND(100.0 * SUM(aki_label) / COUNT(*), 1) AS aki_pct,
    SUM(CASE WHEN hospital_expire_flag = 1 THEN 1 END) AS n_died,
    ROUND(AVG(icu_los_hours)::NUMERIC, 1) AS avg_icu_los_h
FROM cdss_cohort_window;


-- STEP 1-A  cdss_nephrotoxic_rx_raw
-- DROP TABLE IF EXISTS cdss_nephrotoxic_rx_raw CASCADE;

-- [핵심] 신독성 약물 처방 원천 데이터 추출:
-- 코호트 대상 환자에서 약물명 키워드 기반으로 후보 처방을 수집
CREATE TABLE cdss_nephrotoxic_rx_raw AS
SELECT
    p.subject_id, p.drug, p.starttime, p.stoptime,
    p.dose_val_rx, p.dose_unit_rx, p.route
FROM mimiciv_hosp.prescriptions p
WHERE
    p.subject_id IN (SELECT subject_id FROM cdss_cohort_window)
    AND (
        LOWER(p.drug) LIKE '%vancomycin%'   OR LOWER(p.drug) LIKE '%piperacillin%'
     OR LOWER(p.drug) LIKE '%zosyn%'        OR LOWER(p.drug) LIKE '%pip/tazo%'
     OR LOWER(p.drug) LIKE '%pip-tazo%'     OR LOWER(p.drug) LIKE '%gentamicin%'
     OR LOWER(p.drug) LIKE '%tobramycin%'   OR LOWER(p.drug) LIKE '%amikacin%'
     OR LOWER(p.drug) LIKE '%amphotericin%' OR LOWER(p.drug) LIKE '%meropenem%'
     OR LOWER(p.drug) LIKE '%imipenem%'     OR LOWER(p.drug) LIKE '%ertapenem%'
     OR LOWER(p.drug) LIKE '%ketorolac%'    OR LOWER(p.drug) LIKE '%ibuprofen%'
     OR LOWER(p.drug) LIKE '%indomethacin%' OR LOWER(p.drug) LIKE '%diclofenac%'
     OR LOWER(p.drug) LIKE '%lisinopril%'   OR LOWER(p.drug) LIKE '%enalapril%'
     OR LOWER(p.drug) LIKE '%captopril%'    OR LOWER(p.drug) LIKE '%ramipril%'
     OR LOWER(p.drug) LIKE '%losartan%'     OR LOWER(p.drug) LIKE '%valsartan%'
     OR LOWER(p.drug) LIKE '%irbesartan%'   OR LOWER(p.drug) LIKE '%furosemide%'
     OR LOWER(p.drug) LIKE '%lasix%'        OR LOWER(p.drug) LIKE '%hydrochlorothiazide%'
     OR LOWER(p.drug) LIKE '%tacrolimus%'   OR LOWER(p.drug) LIKE '%prograf%'
     OR LOWER(p.drug) LIKE '%cyclosporine%' OR LOWER(p.drug) LIKE '%cyclosporin%'
     OR LOWER(p.drug) LIKE '%metformin%'    OR LOWER(p.drug) LIKE '%pantoprazole%'
     OR LOWER(p.drug) LIKE '%omeprazole%'   OR LOWER(p.drug) LIKE '%esomeprazole%'
    );

CREATE INDEX IF NOT EXISTS cdss_idx_nrx_subject_time
    ON cdss_nephrotoxic_rx_raw (subject_id, starttime);


-- STEP 1-B  cdss_icu_nephrotoxic_rx
-- DROP TABLE IF EXISTS cdss_icu_nephrotoxic_rx CASCADE;

-- [핵심] ICU 구간 약물 노출 피처화:
-- 유효 cutoff 이전 처방만 집계하여 약물별 사용 여부/노출시간/누적용량 산출
CREATE TABLE cdss_icu_nephrotoxic_rx AS
WITH rx_flags AS (
    SELECT
        c.stay_id,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%vancomycin%' THEN 1 ELSE 0 END) AS vancomycin_rx,
        COALESCE(SUM(CASE WHEN LOWER(p.drug) LIKE '%vancomycin%'
            THEN EXTRACT(EPOCH FROM (
                LEAST(COALESCE(p.stoptime, c.effective_cutoff), c.effective_cutoff)
                - p.starttime)) / 3600.0 END), 0) AS vancomycin_exposure_hours,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%piperacillin%' OR LOWER(p.drug) LIKE '%zosyn%'
                   OR LOWER(p.drug) LIKE '%pip/tazo%'     OR LOWER(p.drug) LIKE '%pip-tazo%'
                 THEN 1 ELSE 0 END) AS piptazo_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%gentamicin%'  OR LOWER(p.drug) LIKE '%tobramycin%'
                   OR LOWER(p.drug) LIKE '%amikacin%'
                 THEN 1 ELSE 0 END) AS aminoglycoside_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%amphotericin%' THEN 1 ELSE 0 END) AS amphotericin_b_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%meropenem%'   OR LOWER(p.drug) LIKE '%imipenem%'
                   OR LOWER(p.drug) LIKE '%ertapenem%'
                 THEN 1 ELSE 0 END) AS carbapenem_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%ketorolac%'   THEN 1 ELSE 0 END) AS ketorolac_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%ibuprofen%'   THEN 1 ELSE 0 END) AS ibuprofen_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%ketorolac%'   OR LOWER(p.drug) LIKE '%ibuprofen%'
                   OR LOWER(p.drug) LIKE '%indomethacin%' OR LOWER(p.drug) LIKE '%diclofenac%'
                 THEN 1 ELSE 0 END) AS nsaid_any_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%lisinopril%'  OR LOWER(p.drug) LIKE '%enalapril%'
                   OR LOWER(p.drug) LIKE '%captopril%'   OR LOWER(p.drug) LIKE '%ramipril%'
                 THEN 1 ELSE 0 END) AS ace_inhibitor_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%losartan%'    OR LOWER(p.drug) LIKE '%valsartan%'
                   OR LOWER(p.drug) LIKE '%irbesartan%'
                 THEN 1 ELSE 0 END) AS arb_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%lisinopril%'  OR LOWER(p.drug) LIKE '%enalapril%'
                   OR LOWER(p.drug) LIKE '%captopril%'   OR LOWER(p.drug) LIKE '%losartan%'
                   OR LOWER(p.drug) LIKE '%valsartan%'
                 THEN 1 ELSE 0 END) AS acei_arb_any_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%furosemide%'  OR LOWER(p.drug) LIKE '%lasix%'
                 THEN 1 ELSE 0 END) AS furosemide_rx,
        COALESCE(SUM(CASE WHEN (LOWER(p.drug) LIKE '%furosemide%' OR LOWER(p.drug) LIKE '%lasix%')
                  AND p.dose_val_rx ~ '^[0-9.]+$' THEN p.dose_val_rx::FLOAT END), 0)
                                    AS furosemide_cumulative_mg,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%hydrochlorothiazide%' THEN 1 ELSE 0 END) AS hctz_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%tacrolimus%'  OR LOWER(p.drug) LIKE '%prograf%'
                 THEN 1 ELSE 0 END) AS tacrolimus_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%cyclosporine%' OR LOWER(p.drug) LIKE '%cyclosporin%'
                 THEN 1 ELSE 0 END) AS cyclosporine_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%metformin%'   THEN 1 ELSE 0 END) AS metformin_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%pantoprazole%' OR LOWER(p.drug) LIKE '%omeprazole%'
                   OR LOWER(p.drug) LIKE '%esomeprazole%'
                 THEN 1 ELSE 0 END) AS ppi_rx
    FROM cdss_cohort_window c
    LEFT JOIN cdss_nephrotoxic_rx_raw p
           ON p.subject_id = c.subject_id
          AND p.starttime >= c.icu_intime
          AND p.starttime <  c.effective_cutoff
    GROUP BY c.stay_id
)
SELECT
    cw.stay_id, cw.aki_label, cw.aki_stage, cw.icu_intime, cw.effective_cutoff, cw.is_pseudo_cutoff,
    COALESCE(r.vancomycin_rx,             0) AS vancomycin_rx,
    COALESCE(r.vancomycin_exposure_hours, 0) AS vancomycin_exposure_hours,
    COALESCE(r.piptazo_rx,                0) AS piptazo_rx,
    COALESCE(r.aminoglycoside_rx,         0) AS aminoglycoside_rx,
    COALESCE(r.amphotericin_b_rx,         0) AS amphotericin_b_rx,
    COALESCE(r.carbapenem_rx,             0) AS carbapenem_rx,
    COALESCE(r.ketorolac_rx,              0) AS ketorolac_rx,
    COALESCE(r.ibuprofen_rx,              0) AS ibuprofen_rx,
    COALESCE(r.nsaid_any_rx,              0) AS nsaid_any_rx,
    COALESCE(r.ace_inhibitor_rx,          0) AS ace_inhibitor_rx,
    COALESCE(r.arb_rx,                    0) AS arb_rx,
    COALESCE(r.acei_arb_any_rx,           0) AS acei_arb_any_rx,
    COALESCE(r.furosemide_rx,             0) AS furosemide_rx,
    COALESCE(r.furosemide_cumulative_mg,  0) AS furosemide_cumulative_mg,
    COALESCE(r.hctz_rx,                   0) AS hctz_rx,
    COALESCE(r.tacrolimus_rx,             0) AS tacrolimus_rx,
    COALESCE(r.cyclosporine_rx,           0) AS cyclosporine_rx,
    COALESCE(r.metformin_rx,              0) AS metformin_rx,
    COALESCE(r.ppi_rx,                    0) AS ppi_rx
FROM cdss_cohort_window cw
LEFT JOIN rx_flags r ON cw.stay_id = r.stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_icu_rx_stay ON cdss_icu_nephrotoxic_rx (stay_id);


-- STEP 1-C  cdss_nephrotoxic_combo_risk
-- DROP TABLE IF EXISTS cdss_nephrotoxic_combo_risk CASCADE;

-- [핵심] 약물 조합 위험 피처 생성:
-- 임상적으로 알려진 조합(vanco+piptazo, triple whammy 등)과 burden/risk score 계산
CREATE TABLE cdss_nephrotoxic_combo_risk AS
SELECT
    r.stay_id, r.aki_label,
    r.vancomycin_rx * r.piptazo_rx        AS vanco_piptazo_combo,
    r.vancomycin_rx * r.aminoglycoside_rx AS vanco_aminogly_combo,
    r.vancomycin_rx * r.carbapenem_rx     AS vanco_carbapenem_combo,
    r.nsaid_any_rx  * r.acei_arb_any_rx   AS nsaid_acei_combo,
    CASE WHEN r.nsaid_any_rx = 1 AND r.acei_arb_any_rx = 1 AND r.furosemide_rx = 1
         THEN 1 ELSE 0 END                AS triple_whammy,
    CASE WHEN r.furosemide_cumulative_mg > 200
              AND (r.vancomycin_rx = 1 OR r.aminoglycoside_rx = 1)
         THEN 1 ELSE 0 END                AS diuretic_overload_flag,
    CASE WHEN r.metformin_rx = 1
              AND (r.vancomycin_rx = 1 OR r.nsaid_any_rx = 1)
         THEN 1 ELSE 0 END                AS metformin_risk_flag,
    r.vancomycin_rx + r.aminoglycoside_rx + r.piptazo_rx
    + r.amphotericin_b_rx + r.nsaid_any_rx + r.acei_arb_any_rx + r.tacrolimus_rx
    + CASE WHEN r.furosemide_cumulative_mg > 200 THEN 1 ELSE 0 END
                                          AS nephrotoxic_burden_score,
    (CASE WHEN r.vancomycin_rx = 1 AND r.piptazo_rx = 1 THEN 1 ELSE 0 END)
    + (CASE WHEN r.vancomycin_rx + r.aminoglycoside_rx + r.amphotericin_b_rx >= 2 THEN 1 ELSE 0 END)
    + (CASE WHEN r.vancomycin_exposure_hours > 48 THEN 1 ELSE 0 END)
    + (CASE WHEN r.nsaid_any_rx = 1 AND r.acei_arb_any_rx = 1 THEN 1 ELSE 0 END)
    + (CASE WHEN r.furosemide_cumulative_mg > 200 THEN 1 ELSE 0 END)
                                          AS drug_risk_score,
    r.vancomycin_rx + r.aminoglycoside_rx + r.piptazo_rx
    + r.amphotericin_b_rx + r.nsaid_any_rx + r.acei_arb_any_rx + r.tacrolimus_rx
    + CASE WHEN r.furosemide_cumulative_mg > 200 THEN 1 ELSE 0 END
                                          AS icu_nephrotoxic_count
FROM cdss_icu_nephrotoxic_rx r;

CREATE INDEX IF NOT EXISTS cdss_idx_combo_stay ON cdss_nephrotoxic_combo_risk (stay_id);


-- STEP 2-A  cdss_raw_lab_values
-- DROP TABLE IF EXISTS cdss_raw_lab_values CASCADE;

-- [핵심] Lab 원천값 수집:
-- ICU 입실~유효 cutoff 구간의 주요 신장/대사 관련 검사값만 추출
CREATE TABLE cdss_raw_lab_values AS
SELECT
    c.stay_id, c.effective_cutoff, c.aki_label,
    l.charttime, l.itemid, l.valuenum
FROM cdss_cohort_window c
JOIN mimiciv_hosp.labevents l
      ON  l.subject_id = c.subject_id
      AND l.charttime >= c.icu_intime
      AND l.charttime <= c.effective_cutoff
WHERE l.itemid IN (50912, 51006, 50882, 50971, 51222, 50813)
  AND l.valuenum IS NOT NULL;

CREATE INDEX IF NOT EXISTS cdss_idx_raw_lab_stay ON cdss_raw_lab_values (stay_id, itemid);


-- STEP 2-B  cdss_lab_features
-- DROP TABLE IF EXISTS cdss_lab_features CASCADE;

-- [핵심] Lab 통계 피처 생성:
-- min/max/mean/delta, ratio, 결측 플래그 등 모델 입력용 파생변수 구성
CREATE TABLE cdss_lab_features AS
WITH ranked AS (
    SELECT *,
        LAG(valuenum) OVER (PARTITION BY stay_id, itemid ORDER BY charttime) AS prev_value
    FROM cdss_raw_lab_values
)
SELECT
    stay_id, aki_label,
    MIN(CASE WHEN itemid = 50912 THEN valuenum END) AS cr_min,
    MAX(CASE WHEN itemid = 50912 THEN valuenum END) AS cr_max,
    AVG(CASE WHEN itemid = 50912 THEN valuenum END) AS cr_mean,
    MAX(CASE WHEN itemid = 50912 THEN valuenum END)
    - MIN(CASE WHEN itemid = 50912 THEN valuenum END) AS cr_delta,
    MAX(CASE WHEN itemid = 51006 THEN valuenum END) AS bun_max,
    AVG(CASE WHEN itemid = 51006 THEN valuenum END) AS bun_mean,
    CASE WHEN AVG(CASE WHEN itemid = 50912 THEN valuenum END) > 0
         THEN AVG(CASE WHEN itemid = 51006 THEN valuenum END)
            / AVG(CASE WHEN itemid = 50912 THEN valuenum END)
         ELSE NULL END                              AS bun_cr_ratio,
    MAX(CASE WHEN itemid = 50971 THEN valuenum END) AS potassium_max,
    AVG(CASE WHEN itemid = 50971 THEN valuenum END) AS potassium_mean,
    MIN(CASE WHEN itemid = 50882 THEN valuenum END) AS bicarbonate_min,
    AVG(CASE WHEN itemid = 50882 THEN valuenum END) AS bicarbonate_mean,
    MIN(CASE WHEN itemid = 51222 THEN valuenum END) AS hemoglobin_min,
    AVG(CASE WHEN itemid = 51222 THEN valuenum END) AS hemoglobin_mean,
    MAX(CASE WHEN itemid = 50813 THEN valuenum END) AS lactate_max,
    AVG(CASE WHEN itemid = 50813 THEN valuenum END) AS lactate_mean,
    CASE WHEN COUNT(CASE WHEN itemid = 50912 THEN 1 END) = 0 THEN 1 ELSE 0 END AS cr_missing,
    CASE WHEN COUNT(CASE WHEN itemid = 50813 THEN 1 END) = 0 THEN 1 ELSE 0 END AS lactate_missing
FROM cdss_raw_lab_values
GROUP BY stay_id, aki_label;

CREATE INDEX IF NOT EXISTS cdss_idx_lab_stay ON cdss_lab_features (stay_id);


-- STEP 3-A  cdss_raw_map_values
-- DROP TABLE IF EXISTS cdss_raw_map_values CASCADE;

-- [핵심] MAP 원천 시계열 추출:
-- 허용 범위(이상치 제거) 안의 MAP만 남겨 저관류 시간 계산의 기반 데이터 생성
CREATE TABLE cdss_raw_map_values AS
SELECT c.stay_id, c.effective_cutoff, ce.charttime, ce.valuenum AS map
FROM cdss_cohort_window c
JOIN mimiciv_icu.chartevents ce
      ON  ce.stay_id   = c.stay_id
      AND ce.charttime >= c.icu_intime
      AND ce.charttime <= c.effective_cutoff
WHERE ce.itemid IN (220052, 220181, 225312)
  AND ce.valuenum IS NOT NULL
  AND ce.valuenum BETWEEN 20 AND 300;

CREATE INDEX IF NOT EXISTS cdss_idx_raw_map_stay_time
    ON cdss_raw_map_values (stay_id, charttime);


-- STEP 3-B  cdss_feat_map_ischemia
-- DROP TABLE IF EXISTS cdss_feat_map_ischemia CASCADE;

-- [핵심] MAP 기반 허혈 피처 생성:
-- MAP<65 구간 시간을 누적해 허혈 지속시간 및 임계 초과 플래그 산출
CREATE TABLE cdss_feat_map_ischemia AS
WITH time_ordered AS (
    SELECT stay_id, effective_cutoff, charttime, map,
        LEAD(charttime) OVER (PARTITION BY stay_id ORDER BY charttime) AS next_charttime
    FROM cdss_raw_map_values
),
ischemia_duration AS (
    SELECT stay_id,
        CASE WHEN map < 65 THEN
            EXTRACT(EPOCH FROM (
                LEAST(
                    COALESCE(next_charttime, effective_cutoff),
                    effective_cutoff,
                    charttime + INTERVAL '1 hour'
                ) - charttime
            )) / 3600.0
        ELSE 0 END AS ischemia_hours
    FROM time_ordered
)
SELECT
    stay_id,
    SUM(ischemia_hours) AS map_below65_hours,
    CASE WHEN SUM(ischemia_hours) > 2 THEN 1 ELSE 0 END AS flag_ischemia_over120min
FROM ischemia_duration
GROUP BY stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_map_isch_stay ON cdss_feat_map_ischemia (stay_id);


-- STEP 3-C  cdss_feat_map_summary
-- DROP TABLE IF EXISTS cdss_feat_map_summary CASCADE;

CREATE TABLE cdss_feat_map_summary AS
WITH agg AS (
    SELECT stay_id, MIN(map) AS map_min, AVG(map) AS map_mean
    FROM cdss_raw_map_values GROUP BY stay_id
),
latest AS (
    SELECT DISTINCT ON (stay_id) stay_id, map AS current_map
    FROM cdss_raw_map_values
    ORDER BY stay_id, charttime DESC
)
SELECT a.stay_id, a.map_min, a.map_mean, l.current_map
FROM agg a JOIN latest l ON a.stay_id = l.stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_map_sum_stay ON cdss_feat_map_summary (stay_id);


-- STEP 3-D  cdss_feat_shock_index
-- DROP TABLE IF EXISTS cdss_feat_shock_index CASCADE;

-- [핵심] 쇼크 인덱스 피처 생성:
-- 동일 charttime의 HR/SBP 조합으로 stay 단위 평균 shock_index 계산
CREATE TABLE cdss_feat_shock_index AS
WITH hr_sbp AS (
    SELECT
        c.stay_id, ce.charttime,
        AVG(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END) AS hr,
        AVG(CASE WHEN ce.itemid IN (220179, 220050) THEN ce.valuenum END) AS sbp
    FROM cdss_cohort_window c
    JOIN mimiciv_icu.chartevents ce
          ON  ce.stay_id   = c.stay_id
          AND ce.charttime >= c.icu_intime
          AND ce.charttime <= c.effective_cutoff
    WHERE ce.itemid IN (220045, 220179, 220050)
      AND ((ce.itemid = 220045 AND ce.valuenum BETWEEN 20 AND 250)
        OR (ce.itemid IN (220179, 220050) AND ce.valuenum BETWEEN 40 AND 300))
    GROUP BY c.stay_id, ce.charttime
)
SELECT stay_id, AVG(hr / NULLIF(sbp, 0)) AS shock_index
FROM hr_sbp WHERE hr IS NOT NULL AND sbp IS NOT NULL
GROUP BY stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_shock_stay ON cdss_feat_shock_index (stay_id);


-- STEP 3-E  cdss_feat_vasopressor
-- DROP TABLE IF EXISTS cdss_feat_vasopressor CASCADE;

-- [핵심] 혈압상승제 사용 피처 생성:
-- ICU 구간 inputevents를 기반으로 vasopressor 투여 여부를 이진 플래그화
CREATE TABLE cdss_feat_vasopressor AS
SELECT
    a.stay_id,
    CASE WHEN COUNT(rv.stay_id) > 0 THEN 1 ELSE 0 END AS vasopressor_flag
FROM cdss_cohort_window a
LEFT JOIN (
    SELECT c.stay_id FROM cdss_cohort_window c
    JOIN mimiciv_icu.inputevents ie
          ON  ie.stay_id   = c.stay_id
          AND ie.starttime >= c.icu_intime
          AND ie.starttime <= c.effective_cutoff
    WHERE ie.itemid IN (221906, 221289, 221749, 222315, 221662, 221653)
      AND ie.rate IS NOT NULL
      AND NOT (ie.itemid = 221749 AND ie.rateuom = 'mg/min')
      AND NOT (ie.itemid = 222315 AND ie.rateuom = 'units/min')
) rv ON a.stay_id = rv.stay_id
GROUP BY a.stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_vaso_stay ON cdss_feat_vasopressor (stay_id);


-- STEP 3-F  cdss_ischemic_features
-- DROP TABLE IF EXISTS cdss_ischemic_features CASCADE;

-- [핵심] 허혈/저관류 통합 피처 테이블:
-- MAP 요약, 허혈시간, shock index, vasopressor, 관련 결측 플래그를 한 번에 결합
CREATE TABLE cdss_ischemic_features AS
SELECT
    cw.stay_id, cw.aki_label, cw.aki_stage, cw.prediction_cutoff, cw.hours_to_aki,
    ms.current_map,
    ms.map_mean             AS isch_map_mean,
    ms.map_min              AS isch_map_min,
    COALESCE(mi.map_below65_hours, 0)        AS map_below65_hours,
    COALESCE(mi.flag_ischemia_over120min, 0) AS flag_ischemia_over120min,
    si.shock_index                           AS isch_shock_index,
    COALESCE(vp.vasopressor_flag, 0)         AS vasopressor_flag,
    lf.lactate_max       AS isch_lactate_max,
    lf.hemoglobin_min    AS isch_hemo_min,
    CASE WHEN ms.map_min    IS NULL THEN 1 ELSE 0 END AS map_missing,
    CASE WHEN si.shock_index IS NULL THEN 1 ELSE 0 END AS shock_index_missing,
    CASE WHEN lf.lactate_max IS NULL THEN 1 ELSE 0 END AS isch_lactate_missing,
    CASE WHEN lf.hemoglobin_min IS NULL THEN 1 ELSE 0 END AS hemoglobin_missing
FROM cdss_cohort_window           cw
LEFT JOIN cdss_feat_map_summary   ms ON cw.stay_id = ms.stay_id
LEFT JOIN cdss_feat_map_ischemia  mi ON cw.stay_id = mi.stay_id
LEFT JOIN cdss_feat_shock_index   si ON cw.stay_id = si.stay_id
LEFT JOIN cdss_feat_vasopressor   vp ON cw.stay_id = vp.stay_id
LEFT JOIN cdss_lab_features       lf ON cw.stay_id = lf.stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_isch_stay ON cdss_ischemic_features (stay_id);


-- STEP 4  cdss_rule_score_features
-- DROP TABLE IF EXISTS cdss_rule_score_features CASCADE;

-- [핵심] 룰기반 점수 피처 생성: 체크필요
-- Cr/BUN/eGFR/허혈/MAP 임계치 기반 점수(rule_based_score)와 고위험 플래그 계산
CREATE TABLE cdss_rule_score_features AS
WITH
egfr AS (
    SELECT cw.stay_id,
        CASE
            WHEN cw.gender = 'F' THEN
                CASE WHEN COALESCE(l.cr_mean, l.cr_min) <= 0.7
                    THEN 142.0 * POWER(COALESCE(l.cr_mean,l.cr_min)/0.7, -0.241) * POWER(0.9938, cw.age) * 1.012
                    ELSE 142.0 * POWER(COALESCE(l.cr_mean,l.cr_min)/0.7, -1.200) * POWER(0.9938, cw.age) * 1.012
                END
            ELSE
                CASE WHEN COALESCE(l.cr_mean, l.cr_min) <= 0.9
                    THEN 142.0 * POWER(COALESCE(l.cr_mean,l.cr_min)/0.9, -0.302) * POWER(0.9938, cw.age)
                    ELSE 142.0 * POWER(COALESCE(l.cr_mean,l.cr_min)/0.9, -1.200) * POWER(0.9938, cw.age)
                END
        END AS egfr_ckdepi
    FROM cdss_cohort_window cw
    JOIN cdss_lab_features  l ON cw.stay_id = l.stay_id
    WHERE COALESCE(l.cr_mean, l.cr_min) IS NOT NULL
      AND COALESCE(l.cr_mean, l.cr_min) > 0
),
threshold_flags AS (
    SELECT
        cw.stay_id,
        CASE WHEN COALESCE(l.cr_max, l.cr_mean) > 1.5  THEN 1 ELSE 0 END AS flag_cr,
        COALESCE(l.cr_max, l.cr_mean)                   AS val_cr,
        CASE WHEN l.bun_max > 30                        THEN 1 ELSE 0 END AS flag_bun,
        l.bun_max                                       AS val_bun,
        CASE WHEN e.egfr_ckdepi < 45                    THEN 1 ELSE 0 END AS flag_egfr,
        e.egfr_ckdepi                                   AS val_egfr,
        CASE WHEN COALESCE(isch.map_below65_hours,0) > 2 THEN 1 ELSE 0 END AS flag_ischemia,
        COALESCE(isch.map_below65_hours, 0) * 60        AS val_ischemia_min,
        CASE WHEN COALESCE(isch.isch_map_min,999) < 65  THEN 1 ELSE 0 END AS flag_map,
        isch.isch_map_min                               AS val_map
    FROM cdss_cohort_window     cw
    LEFT JOIN cdss_lab_features      l    ON cw.stay_id = l.stay_id
    LEFT JOIN egfr                               e    ON cw.stay_id = e.stay_id
    LEFT JOIN cdss_ischemic_features isch ON cw.stay_id = isch.stay_id
)
SELECT
    cw.stay_id, cw.aki_label,
    e.egfr_ckdepi,
    tf.val_cr, tf.val_bun, tf.val_egfr, tf.val_ischemia_min, tf.val_map,
    tf.flag_cr, tf.flag_bun, tf.flag_egfr, tf.flag_ischemia, tf.flag_map,
    tf.flag_cr       * 30 AS score_cr,
    tf.flag_bun      * 20 AS score_bun,
    tf.flag_egfr     * 20 AS score_egfr,
    tf.flag_ischemia * 15 AS score_ischemia,
    tf.flag_map      * 15 AS score_map,
    tf.flag_cr*30 + tf.flag_bun*20 + tf.flag_egfr*20
    + tf.flag_ischemia*15 + tf.flag_map*15 AS rule_based_score,
    CASE WHEN (tf.flag_cr*30+tf.flag_bun*20+tf.flag_egfr*20
               +tf.flag_ischemia*15+tf.flag_map*15) >= 70
         THEN 1 ELSE 0 END AS high_risk_flag
FROM cdss_cohort_window cw
LEFT JOIN egfr            e  ON cw.stay_id = e.stay_id
LEFT JOIN threshold_flags tf ON cw.stay_id = tf.stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_rule_score_stay ON cdss_rule_score_features (stay_id);


-- STEP 5  cdss_master_features
-- DROP TABLE IF EXISTS cdss_master_features CASCADE;

-- [핵심] 최종 마스터 피처 생성:
-- 코호트 + 약물 + Lab + 허혈 + 룰기반 점수를 통합한 학습용 단일 테이블
CREATE TABLE cdss_master_features AS
SELECT
    cw.stay_id, cw.subject_id, cw.hadm_id,
    cw.age, cw.gender, cw.first_careunit,
    cw.icu_intime, cw.icu_outtime, cw.icu_los_hours,
    cw.aki_label, cw.aki_stage,
    cw.prediction_cutoff, cw.effective_cutoff,
    cw.hours_to_aki, cw.is_pseudo_cutoff,
    cw.hospital_expire_flag, cw.competed_with_death, cw.hours_to_death,
    rx.vancomycin_rx, rx.vancomycin_exposure_hours, rx.piptazo_rx,
    rx.aminoglycoside_rx, rx.amphotericin_b_rx, rx.carbapenem_rx,
    rx.ketorolac_rx, rx.nsaid_any_rx,
    rx.ace_inhibitor_rx, rx.arb_rx, rx.acei_arb_any_rx,
    rx.furosemide_rx, rx.furosemide_cumulative_mg,
    rx.tacrolimus_rx, rx.cyclosporine_rx, rx.metformin_rx, rx.ppi_rx,
    cb.vanco_piptazo_combo, cb.vanco_aminogly_combo, cb.vanco_carbapenem_combo,
    cb.nsaid_acei_combo, cb.triple_whammy, cb.diuretic_overload_flag,
    cb.nephrotoxic_burden_score, cb.drug_risk_score,
    lf.cr_min, lf.cr_max, lf.cr_mean, lf.cr_delta,
    lf.bun_max, lf.bun_mean, lf.bun_cr_ratio,
    lf.potassium_max, lf.potassium_mean,
    lf.bicarbonate_min, lf.bicarbonate_mean,
    lf.hemoglobin_min, lf.hemoglobin_mean,
    lf.lactate_max, lf.lactate_mean,
    lf.cr_missing, lf.lactate_missing,
    rs.egfr_ckdepi,
    isch.current_map, isch.isch_map_mean, isch.isch_map_min,
    isch.map_below65_hours, isch.flag_ischemia_over120min,
    isch.isch_shock_index, isch.vasopressor_flag,
    isch.isch_lactate_max, isch.isch_hemo_min,
    isch.map_missing, isch.shock_index_missing,
    isch.isch_lactate_missing, isch.hemoglobin_missing,
    rs.val_cr, rs.val_bun, rs.val_egfr, rs.val_ischemia_min, rs.val_map,
    rs.flag_cr, rs.flag_bun, rs.flag_egfr, rs.flag_ischemia, rs.flag_map,
    rs.score_cr, rs.score_bun, rs.score_egfr, rs.score_ischemia, rs.score_map,
    rs.rule_based_score, rs.high_risk_flag,
    COALESCE(cb.nephrotoxic_burden_score, 0)
    + COALESCE(rs.flag_cr,   0) * 2
    + COALESCE(rs.flag_egfr, 0) * 2
    + COALESCE(isch.vasopressor_flag, 0) AS total_nephrotoxic_burden
FROM cdss_cohort_window               cw
LEFT JOIN cdss_icu_nephrotoxic_rx     rx   ON cw.stay_id = rx.stay_id
LEFT JOIN cdss_nephrotoxic_combo_risk cb   ON cw.stay_id = cb.stay_id
LEFT JOIN cdss_lab_features           lf   ON cw.stay_id = lf.stay_id
LEFT JOIN cdss_ischemic_features      isch ON cw.stay_id = isch.stay_id
LEFT JOIN cdss_rule_score_features    rs   ON cw.stay_id = rs.stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_master_stay     ON cdss_master_features (stay_id);
CREATE INDEX IF NOT EXISTS cdss_idx_master_label    ON cdss_master_features (aki_label);
CREATE INDEX IF NOT EXISTS cdss_idx_master_highrisk ON cdss_master_features (high_risk_flag);


--

--  STEP 7-A  cdss_raw_vital_values
-- DROP TABLE IF EXISTS cdss_raw_vital_values CASCADE;

-- [핵심] 활력징후 원천값 추출:
-- HR/RR/SBP/체온/SpO2를 유효 관찰창 내에서 정제된 범위로 수집
CREATE TABLE cdss_raw_vital_values AS
SELECT
    c.stay_id, c.icu_intime, c.effective_cutoff,
    ce.charttime, ce.itemid, ce.valuenum
FROM cdss_cohort_window c
JOIN mimiciv_icu.chartevents ce
      ON  ce.stay_id   = c.stay_id
      AND ce.charttime >= c.icu_intime
      AND ce.charttime <= c.effective_cutoff
WHERE ce.itemid IN (220045, 220210, 224690, 220179, 220050, 223761, 223762, 220277)
  AND ce.valuenum IS NOT NULL
  AND (
        (ce.itemid = 220045                  AND ce.valuenum BETWEEN 20  AND 250)
     OR (ce.itemid IN (220210, 224690)       AND ce.valuenum BETWEEN 4   AND 60)
     OR (ce.itemid IN (220179, 220050)       AND ce.valuenum BETWEEN 40  AND 300)
     OR (ce.itemid = 223761                  AND ce.valuenum BETWEEN 77  AND 113)
     OR (ce.itemid = 223762                  AND ce.valuenum BETWEEN 25  AND 45)
     OR (ce.itemid = 220277                  AND ce.valuenum BETWEEN 50  AND 100)
  );

CREATE INDEX IF NOT EXISTS cdss_idx_raw_vital_stay_item
    ON cdss_raw_vital_values (stay_id, itemid);


-- ? STEP 7-B  cdss_feat_vitals
-- DROP TABLE IF EXISTS cdss_feat_vitals CASCADE;

-- [핵심] 활력징후 통계 피처 생성:
-- stay 단위 min/max/mean 요약값으로 모델 입력 피처 구성
CREATE TABLE cdss_feat_vitals AS
SELECT
    stay_id,
    MIN(CASE WHEN itemid = 220045 THEN valuenum END)              AS hr_min,
    MAX(CASE WHEN itemid = 220045 THEN valuenum END)              AS hr_max,
    AVG(CASE WHEN itemid = 220045 THEN valuenum END)              AS hr_mean,
    MAX(CASE WHEN itemid IN (220210, 224690) THEN valuenum END)   AS rr_max,
    AVG(CASE WHEN itemid IN (220210, 224690) THEN valuenum END)   AS rr_mean,
    MIN(CASE WHEN itemid IN (220179, 220050) THEN valuenum END)   AS sbp_min,
    AVG(CASE WHEN itemid IN (220179, 220050) THEN valuenum END)   AS sbp_mean,
    MIN(CASE
            WHEN itemid = 223762 THEN valuenum
            WHEN itemid = 223761 THEN (valuenum - 32.0) * 5.0 / 9.0
        END)                                                      AS temp_min,
    AVG(CASE
            WHEN itemid = 223762 THEN valuenum
            WHEN itemid = 223761 THEN (valuenum - 32.0) * 5.0 / 9.0
        END)                                                      AS temp_mean,
    MIN(CASE WHEN itemid = 220277 THEN valuenum END)              AS spo2_min,
    AVG(CASE WHEN itemid = 220277 THEN valuenum END)              AS spo2_mean
FROM cdss_raw_vital_values
GROUP BY stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_feat_vital_stay
    ON cdss_feat_vitals (stay_id);


-- ? STEP 7-C  cdss_feat_urine_fluid
-- DROP TABLE IF EXISTS cdss_feat_urine_fluid CASCADE;

-- [핵심] 소변량/수액 균형 피처 생성:
-- 6h/24h/전체 소변량, 수액량, fluid balance, urine 결측 여부를 계산
CREATE TABLE cdss_feat_urine_fluid AS
WITH urine_raw AS (
    SELECT
        c.stay_id, c.icu_intime, c.effective_cutoff,
        oe.charttime,
        oe.value AS urine_ml
    FROM cdss_cohort_window c
    JOIN mimiciv_icu.outputevents oe
          ON  oe.stay_id   = c.stay_id
          AND oe.charttime >= c.icu_intime
          AND oe.charttime <= c.effective_cutoff
    WHERE oe.itemid IN (226559,226560,226561,226584,226563,226564,226565,226557,226558,226567)
      AND oe.value BETWEEN 0 AND 1500
),
urine_agg AS (
    SELECT
        stay_id,
        SUM(CASE WHEN charttime <= icu_intime + INTERVAL '6 hours'
                 THEN urine_ml ELSE 0 END)  AS urine_6h,
        SUM(CASE WHEN charttime <= icu_intime + INTERVAL '24 hours'
                 THEN urine_ml ELSE 0 END)  AS urine_24h,
        SUM(urine_ml)                       AS urine_total
    FROM urine_raw
    GROUP BY stay_id
),
urine_hourly AS (
    SELECT DISTINCT stay_id, DATE_TRUNC('hour', charttime) AS hour_bucket
    FROM urine_raw WHERE urine_ml > 0
),
zero_ratio AS (
    SELECT
        cw.stay_id,
        1.0 - (
            COUNT(uh.hour_bucket)::FLOAT
            / NULLIF(EXTRACT(EPOCH FROM (cw.effective_cutoff - cw.icu_intime)) / 3600.0, 0)
        ) AS urine_zero_ratio
    FROM cdss_cohort_window cw
    LEFT JOIN urine_hourly uh ON cw.stay_id = uh.stay_id
    GROUP BY cw.stay_id, cw.effective_cutoff, cw.icu_intime
),
fluid_raw AS (
    SELECT
        c.stay_id, c.icu_intime,
        ie.starttime,
        ie.amount AS fluid_ml
    FROM cdss_cohort_window c
    JOIN mimiciv_icu.inputevents ie
          ON  ie.stay_id   = c.stay_id
          AND ie.starttime >= c.icu_intime
          AND ie.starttime <= c.effective_cutoff
    WHERE ie.amountuom = 'mL'
      AND ie.amount BETWEEN 0 AND 5000
),
fluid_agg AS (
    SELECT
        stay_id,
        SUM(CASE WHEN starttime <= icu_intime + INTERVAL '24 hours'
                 THEN fluid_ml ELSE 0 END)  AS fluid_24h,
        SUM(fluid_ml)                       AS fluid_total
    FROM fluid_raw
    GROUP BY stay_id
)
SELECT
    cw.stay_id,
    COALESCE(ua.urine_6h,    0)                                 AS urine_6h,
    COALESCE(ua.urine_24h,   0)                                 AS urine_24h,
    COALESCE(ua.urine_total, 0)                                 AS urine_total,
    GREATEST(0.0, LEAST(1.0, COALESCE(zr.urine_zero_ratio, 1.0))) AS urine_zero_ratio,
    COALESCE(fa.fluid_24h,   0)                                 AS fluid_24h,
    COALESCE(fa.fluid_total, 0)                                 AS fluid_total,
    COALESCE(fa.fluid_24h,   0) - COALESCE(ua.urine_24h,   0)  AS fluid_balance_24h,
    COALESCE(fa.fluid_total, 0) - COALESCE(ua.urine_total, 0)  AS fluid_balance_total,
    CASE WHEN ua.stay_id IS NULL THEN 1 ELSE 0 END              AS urine_missing
FROM cdss_cohort_window    cw
LEFT JOIN urine_agg ua  ON cw.stay_id = ua.stay_id
LEFT JOIN zero_ratio zr ON cw.stay_id = zr.stay_id
LEFT JOIN fluid_agg  fa ON cw.stay_id = fa.stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_urine_fluid_stay
    ON cdss_feat_urine_fluid (stay_id);


-- ? STEP 7-D  cdss_master_features ALTER TABLE + UPDATE

ALTER TABLE cdss_master_features
    ADD COLUMN IF NOT EXISTS hr_min               FLOAT,
    ADD COLUMN IF NOT EXISTS hr_max               FLOAT,
    ADD COLUMN IF NOT EXISTS hr_mean              FLOAT,
    ADD COLUMN IF NOT EXISTS rr_max               FLOAT,
    ADD COLUMN IF NOT EXISTS rr_mean              FLOAT,
    ADD COLUMN IF NOT EXISTS sbp_min              FLOAT,
    ADD COLUMN IF NOT EXISTS sbp_mean             FLOAT,
    ADD COLUMN IF NOT EXISTS temp_min             FLOAT,
    ADD COLUMN IF NOT EXISTS temp_mean            FLOAT,
    ADD COLUMN IF NOT EXISTS spo2_min             FLOAT,
    ADD COLUMN IF NOT EXISTS spo2_mean            FLOAT,
    ADD COLUMN IF NOT EXISTS urine_6h             FLOAT,
    ADD COLUMN IF NOT EXISTS urine_24h            FLOAT,
    ADD COLUMN IF NOT EXISTS urine_total          FLOAT,
    ADD COLUMN IF NOT EXISTS urine_zero_ratio     FLOAT,
    ADD COLUMN IF NOT EXISTS fluid_24h            FLOAT,
    ADD COLUMN IF NOT EXISTS fluid_total          FLOAT,
    ADD COLUMN IF NOT EXISTS fluid_balance_24h    FLOAT,
    ADD COLUMN IF NOT EXISTS fluid_balance_total  FLOAT,
    ADD COLUMN IF NOT EXISTS urine_missing        SMALLINT DEFAULT 1;

UPDATE cdss_master_features m
SET
    hr_min    = v.hr_min,    hr_max   = v.hr_max,   hr_mean  = v.hr_mean,
    rr_max    = v.rr_max,    rr_mean  = v.rr_mean,
    sbp_min   = v.sbp_min,   sbp_mean = v.sbp_mean,
    temp_min  = v.temp_min,  temp_mean= v.temp_mean,
    spo2_min  = v.spo2_min,  spo2_mean= v.spo2_mean
FROM cdss_feat_vitals v
WHERE m.stay_id = v.stay_id;

UPDATE cdss_master_features m
SET
    urine_6h            = u.urine_6h,
    urine_24h           = u.urine_24h,
    urine_total         = u.urine_total,
    urine_zero_ratio    = u.urine_zero_ratio,
    fluid_24h           = u.fluid_24h,
    fluid_total         = u.fluid_total,
    fluid_balance_24h   = u.fluid_balance_24h,
    fluid_balance_total = u.fluid_balance_total,
    urine_missing       = u.urine_missing
FROM cdss_feat_urine_fluid u
WHERE m.stay_id = u.stay_id;

-- STEP 7
SELECT
    COUNT(*)                                         AS n_total,
    COUNT(hr_mean)                                   AS n_hr_available,
    ROUND(100.0 * COUNT(hr_mean)   / COUNT(*), 1)    AS hr_coverage_pct,
    COUNT(spo2_min)                                  AS n_spo2_available,
    ROUND(100.0 * COUNT(spo2_min)  / COUNT(*), 1)    AS spo2_coverage_pct,
    SUM(1 - urine_missing)                           AS n_urine_available,
    ROUND(100.0 * SUM(1 - urine_missing) / COUNT(*), 1) AS urine_coverage_pct,
    ROUND(AVG(hr_mean)::NUMERIC,    1)               AS avg_hr,
    ROUND(AVG(urine_24h)::NUMERIC,  0)               AS avg_urine_24h_ml,
    ROUND(AVG(fluid_balance_24h)::NUMERIC, 0)        AS avg_fluid_balance_24h_ml
FROM cdss_master_features;



-- STEP 6-A  stg_nlp_keyword_features
-- DROP TABLE IF EXISTS stg_nlp_keyword_features CASCADE;

-- [핵심] NLP 키워드 스테이징:
-- 외부에서 생성한 텍스트 키워드 피처를 적재하기 위한 입력 테이블
CREATE TABLE stg_nlp_keyword_features (
    stay_id             BIGINT      PRIMARY KEY,
    nlp_text_combined   TEXT,
    kw_oliguria         SMALLINT    DEFAULT 0,
    kw_anuria           SMALLINT    DEFAULT 0,
    kw_edema            SMALLINT    DEFAULT 0,
    kw_hydronephrosis   SMALLINT    DEFAULT 0,
    kw_aki_mention      SMALLINT    DEFAULT 0,
    kw_renal_abnormal   SMALLINT    DEFAULT 0,
    kw_fluid_overload   SMALLINT    DEFAULT 0,
    imported_at         TIMESTAMPTZ DEFAULT NOW()
);
-- \\COPY stg_nlp_keyword_features
--   (stay_id, nlp_text_combined, kw_oliguria, kw_anuria, kw_edema,
--    kw_hydronephrosis, kw_aki_mention, kw_renal_abnormal, kw_fluid_overload)
-- FROM 'kidney_nlp/nlp_keyword_features.csv' CSV HEADER ENCODING 'UTF8';


-- STEP 6-B  stg_radiology_nlp_text
-- DROP TABLE IF EXISTS stg_radiology_nlp_text CASCADE;

-- [핵심] 영상의학 텍스트 스테이징:
-- findings/impression 원문을 적재해 방사선 NLP 규칙 피처 추출에 사용
CREATE TABLE stg_radiology_nlp_text (
    note_id     TEXT        NOT NULL,
    subject_id  BIGINT      NOT NULL,
    hadm_id     BIGINT,
    stay_id     BIGINT,
    charttime   TIMESTAMPTZ,
    findings    TEXT,
    impression  TEXT,
    imported_at TIMESTAMPTZ DEFAULT NOW()
);
-- \\COPY stg_radiology_nlp_text
--   (note_id, subject_id, hadm_id, stay_id, charttime, findings, impression)
-- FROM 'kidney_nlp/radiology_nlp_text.csv' CSV HEADER ENCODING 'UTF8' NULL '';

CREATE INDEX IF NOT EXISTS cdss_idx_rad_stay    ON stg_radiology_nlp_text (stay_id);
CREATE INDEX IF NOT EXISTS cdss_idx_rad_subject ON stg_radiology_nlp_text (subject_id);
CREATE INDEX IF NOT EXISTS cdss_idx_rad_time    ON stg_radiology_nlp_text (charttime);


-- STEP 6-C  cdss_nlp_keyword_raw
-- DROP TABLE IF EXISTS cdss_nlp_keyword_raw CASCADE;

CREATE TABLE cdss_nlp_keyword_raw AS
SELECT
    cw.stay_id, cw.effective_cutoff, cw.aki_label,
    COALESCE(n.kw_oliguria,       0) AS kw_oliguria,
    COALESCE(n.kw_anuria,         0) AS kw_anuria,
    COALESCE(n.kw_edema,          0) AS kw_edema,
    COALESCE(n.kw_hydronephrosis, 0) AS kw_hydronephrosis,
    COALESCE(n.kw_aki_mention,    0) AS kw_aki_mention,
    COALESCE(n.kw_renal_abnormal, 0) AS kw_renal_abnormal,
    COALESCE(n.kw_fluid_overload, 0) AS kw_fluid_overload,
    CASE WHEN n.stay_id IS NULL THEN 1 ELSE 0 END AS nlp_missing,
    COALESCE(n.kw_oliguria,0) + COALESCE(n.kw_anuria,0) + COALESCE(n.kw_edema,0)
    + COALESCE(n.kw_hydronephrosis,0) + COALESCE(n.kw_aki_mention,0)
    + COALESCE(n.kw_renal_abnormal,0) + COALESCE(n.kw_fluid_overload,0) AS nlp_keyword_score
FROM cdss_cohort_window            cw
LEFT JOIN stg_nlp_keyword_features n ON cw.stay_id = n.stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_nlp_kw_stay ON cdss_nlp_keyword_raw (stay_id);


-- STEP 6-D  cdss_nlp_radiology_extra
-- DROP TABLE IF EXISTS cdss_nlp_radiology_extra CASCADE;

CREATE TABLE cdss_nlp_radiology_extra AS
WITH filtered_notes AS (
    SELECT r.stay_id, r.charttime,
        LOWER(COALESCE(r.findings,'') || ' ' || COALESCE(r.impression,'')) AS note_text
    FROM stg_radiology_nlp_text r
    JOIN cdss_cohort_window cw ON r.stay_id = cw.stay_id
    WHERE r.stay_id IS NOT NULL
      AND r.charttime <= cw.effective_cutoff
      AND r.charttime >= cw.icu_intime
      AND (r.findings IS NOT NULL OR r.impression IS NOT NULL)
),
note_flags AS (
    SELECT stay_id, charttime,
        CASE WHEN note_text ~* '\\mpulmonary\\s+edema\\M|\\mpulmonary\\s+congestion\\M|\\mcongestive\\s+heart\\s+failure\\M'
             THEN 1 ELSE 0 END AS kw_pulmonary_edema,
        CASE WHEN note_text ~* '\\mpleural\\s+effusion\\M|\\mpleural\\s+fluid\\M'
             THEN 1 ELSE 0 END AS kw_pleural_effusion,
        CASE WHEN note_text ~* '\\mascit[eis]\\M|\\mperitoneal\\s+fluid\\M|\\mfree\\s+fluid.*abdomen\\M'
             THEN 1 ELSE 0 END AS kw_ascites,
        CASE WHEN note_text ~* '\\mcontrast\\M|\\miodinated\\s+contrast\\M|\\mgadolinium\\M|\\mnephropathy\\M'
             THEN 1 ELSE 0 END AS kw_contrast_agent,
        CASE WHEN note_text ~* '\\mhydronephrosis\\M|\\mhydroureter\\M|\\mrenal\\s+obstruction\\M|\\mureteral\\s+obstruction\\M'
             THEN 1 ELSE 0 END AS kw_rad_hydronephrosis,
        CASE WHEN note_text ~* '\\mrenal\\s+calcul[ui]\\M|\\mkidney\\s+stone\\M|\\mnephrolithiasis\\M|\\murolithiasis\\M|\\mureteral\\s+calcul[ui]\\M'
             THEN 1 ELSE 0 END AS kw_renal_calculus,
        CASE WHEN note_text ~* '\\mcardiomegaly\\M|\\mcardiac\\s+enlargement\\M|\\menlarged\\s+(cardiac\\s+)?silhouette\\M'
             THEN 1 ELSE 0 END AS kw_cardiomegaly,
        CASE WHEN note_text ~* '\\mfoley\\M|\\murinary\\s+(catheter|drain)\\M|\\mbladder\\s+catheter\\M'
             THEN 1 ELSE 0 END AS kw_foley_catheter,
        CASE WHEN note_text ~* '\\mrenal\\s+failure\\M|\\macute\\s+kidney\\s+injury\\M|\\m\\baki\\b\\M|\\mrenal\\s+insufficiency\\M'
             THEN 1 ELSE 0 END AS kw_rad_aki_mention
    FROM filtered_notes
)
SELECT
    stay_id,
    MAX(kw_pulmonary_edema)    AS kw_pulmonary_edema,
    MAX(kw_pleural_effusion)   AS kw_pleural_effusion,
    MAX(kw_ascites)            AS kw_ascites,
    MAX(kw_contrast_agent)     AS kw_contrast_agent,
    MAX(kw_rad_hydronephrosis) AS kw_rad_hydronephrosis,
    MAX(kw_renal_calculus)     AS kw_renal_calculus,
    MAX(kw_cardiomegaly)       AS kw_cardiomegaly,
    MAX(kw_foley_catheter)     AS kw_foley_catheter,
    MAX(kw_rad_aki_mention)    AS kw_rad_aki_mention,
    COUNT(*)                   AS rad_report_count,
    0                          AS rad_text_missing
FROM note_flags
GROUP BY stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_nlp_rad_stay ON cdss_nlp_radiology_extra (stay_id);


-- STEP 6-E  cdss_nlp_features
-- DROP TABLE IF EXISTS cdss_nlp_features CASCADE;

-- [핵심] NLP 통합 피처 생성:
-- 키워드/방사선 텍스트 플래그, 누락 여부, 직접 신장위험/체액부담 플래그 통합
CREATE TABLE cdss_nlp_features AS
SELECT
    cw.stay_id, cw.aki_label, cw.effective_cutoff,
    COALESCE(k.kw_oliguria,       0) AS kw_oliguria,
    COALESCE(k.kw_anuria,         0) AS kw_anuria,
    COALESCE(k.kw_edema,          0) AS kw_edema,
    COALESCE(k.kw_hydronephrosis, 0) AS kw_hydronephrosis,
    COALESCE(k.kw_aki_mention,    0) AS kw_aki_mention,
    COALESCE(k.kw_renal_abnormal, 0) AS kw_renal_abnormal,
    COALESCE(k.kw_fluid_overload, 0) AS kw_fluid_overload,
    COALESCE(k.nlp_keyword_score, 0) AS nlp_keyword_score,
    COALESCE(r.kw_pulmonary_edema,    0) AS kw_pulmonary_edema,
    COALESCE(r.kw_pleural_effusion,   0) AS kw_pleural_effusion,
    COALESCE(r.kw_ascites,            0) AS kw_ascites,
    COALESCE(r.kw_contrast_agent,     0) AS kw_contrast_agent,
    COALESCE(r.kw_rad_hydronephrosis, 0) AS kw_rad_hydronephrosis,
    COALESCE(r.kw_renal_calculus,     0) AS kw_renal_calculus,
    COALESCE(r.kw_cardiomegaly,       0) AS kw_cardiomegaly,
    COALESCE(r.kw_foley_catheter,     0) AS kw_foley_catheter,
    COALESCE(r.kw_rad_aki_mention,    0) AS kw_rad_aki_mention,
    COALESCE(r.rad_report_count,      0) AS rad_report_count,
    COALESCE(k.nlp_missing,      1)      AS nlp_missing,
    COALESCE(r.rad_text_missing, 1)      AS rad_text_missing,
    CASE WHEN COALESCE(k.kw_hydronephrosis,0)=1 OR COALESCE(r.kw_rad_hydronephrosis,0)=1
           OR COALESCE(r.kw_renal_calculus,0)=1 OR COALESCE(k.kw_aki_mention,0)=1
           OR COALESCE(r.kw_rad_aki_mention,0)=1 THEN 1 ELSE 0 END AS nlp_direct_renal_flag,
    CASE WHEN (COALESCE(k.kw_edema,0) + COALESCE(k.kw_fluid_overload,0)
              + COALESCE(r.kw_pulmonary_edema,0) + COALESCE(r.kw_pleural_effusion,0)
              + COALESCE(r.kw_ascites,0)) >= 2 THEN 1 ELSE 0 END   AS nlp_fluid_burden_flag
FROM cdss_cohort_window            cw
LEFT JOIN cdss_nlp_keyword_raw     k ON cw.stay_id = k.stay_id
LEFT JOIN cdss_nlp_radiology_extra r ON cw.stay_id = r.stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_nlp_feat_stay ON cdss_nlp_features (stay_id);


-- STEP 6-F  cdss_master_features NLP 而щ읆 異붽?

ALTER TABLE cdss_master_features
    ADD COLUMN IF NOT EXISTS kw_oliguria            SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_anuria              SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_edema               SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_hydronephrosis      SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_aki_mention         SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_renal_abnormal      SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_fluid_overload      SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_pulmonary_edema     SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_pleural_effusion    SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_ascites             SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_contrast_agent      SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_rad_hydronephrosis  SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_renal_calculus      SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_cardiomegaly        SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_foley_catheter      SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_rad_aki_mention     SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS nlp_keyword_score      SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS rad_report_count       INT      DEFAULT 0,
    ADD COLUMN IF NOT EXISTS nlp_missing            SMALLINT DEFAULT 1,
    ADD COLUMN IF NOT EXISTS rad_text_missing       SMALLINT DEFAULT 1,
    ADD COLUMN IF NOT EXISTS nlp_direct_renal_flag  SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS nlp_fluid_burden_flag  SMALLINT DEFAULT 0;

UPDATE cdss_master_features m
SET
    kw_oliguria           = n.kw_oliguria,
    kw_anuria             = n.kw_anuria,
    kw_edema              = n.kw_edema,
    kw_hydronephrosis     = n.kw_hydronephrosis,
    kw_aki_mention        = n.kw_aki_mention,
    kw_renal_abnormal     = n.kw_renal_abnormal,
    kw_fluid_overload     = n.kw_fluid_overload,
    kw_pulmonary_edema    = n.kw_pulmonary_edema,
    kw_pleural_effusion   = n.kw_pleural_effusion,
    kw_ascites            = n.kw_ascites,
    kw_contrast_agent     = n.kw_contrast_agent,
    kw_rad_hydronephrosis = n.kw_rad_hydronephrosis,
    kw_renal_calculus     = n.kw_renal_calculus,
    kw_cardiomegaly       = n.kw_cardiomegaly,
    kw_foley_catheter     = n.kw_foley_catheter,
    kw_rad_aki_mention    = n.kw_rad_aki_mention,
    nlp_keyword_score     = n.nlp_keyword_score,
    rad_report_count      = n.rad_report_count,
    nlp_missing           = n.nlp_missing,
    rad_text_missing      = n.rad_text_missing,
    nlp_direct_renal_flag = n.nlp_direct_renal_flag,
    nlp_fluid_burden_flag = n.nlp_fluid_burden_flag
FROM cdss_nlp_features n
WHERE m.stay_id = n.stay_id;


SELECT
    COUNT(*)                                          AS n_total,
    SUM(aki_label)                                    AS n_aki,
    ROUND(100.0*SUM(aki_label)/COUNT(*),1)             AS aki_pct,
    ROUND(AVG(rule_based_score)::NUMERIC,1)            AS avg_rule_score,
    SUM(high_risk_flag)                               AS n_high_risk,
    SUM(vanco_piptazo_combo)                          AS n_vanco_pip,
    SUM(triple_whammy)                                AS n_triple_whammy,
    SUM(competed_with_death)                          AS n_competing_risk,
    COUNT(hr_mean)                                    AS n_hr_available,
    ROUND(100.0*COUNT(hr_mean)/COUNT(*),1)             AS hr_coverage_pct,
    SUM(1 - urine_missing)                            AS n_urine_available,
    ROUND(100.0*SUM(1-urine_missing)/COUNT(*),1)       AS urine_coverage_pct,
    ROUND(AVG(urine_24h)::NUMERIC, 0)                 AS avg_urine_24h_ml,
    -- STEP 6 NLP
    SUM(1 - nlp_missing)                              AS n_nlp_available,
    ROUND(100.0*SUM(1-nlp_missing)/COUNT(*),1)         AS nlp_coverage_pct,
    SUM(kw_edema)                                     AS n_kw_edema,
    SUM(kw_fluid_overload)                            AS n_kw_fluid_overload,
    SUM(nlp_direct_renal_flag)                        AS n_direct_renal,
    SUM(nlp_fluid_burden_flag)                        AS n_fluid_burden
FROM cdss_master_features;
	
# Renal AKI CDSS

신장 이식 환자 중환자실 **급성 신손상(AKI) 조기 예측** 및 임상 의사 결정 지원 시스템 (CDSS)

---

## 개요

XGBoost 기반 AI 모델을 활용하여 **6시간 / 12시간 / 24시간 / 48시간** 후 AKI 발생 확률을 실시간으로 예측하고, KDIGO 2012 기준에 따른 AKI 병기 분류 및 임상 권고를 제공하는 Streamlit 웹 애플리케이션입니다.

---

## 주요 기능

### 인증
- **로그인** — 사번 / 비밀번호 기반 인증, 로그인 상태 유지
- **회원가입** — 신규 사원 등록 (성명, 사번, 비밀번호, 주민등록번호, 소속 부서, 직급 등)

### 홈
- 오늘의 입원 환자 · 중환자실 재실 · 신규 원무 등록 · CDSS 활성 알림 통계
- CDSS 대시보드 및 원무 시스템 바로가기
- 최근 공지사항, 나의 업무 현황

### CDSS 대시보드
| 항목 | 설명 |
|---|---|
| 환자 헤더바 | 이름, 나이, 등록번호, 병동, 알레르기, 활력 징후 상태 |
| 활력 징후 추이 | SBP / 심박수 / 체온 시계열 차트 (6h / 12h / 24h 선택) |
| **AKI 다중 시간대 예측** | **6h / 12h / 24h / 48h** 발생 확률 카드 + 막대 차트 |
| 위험도 게이지 | 종합 위험 · 낙상 위험 · 욕창 위험 (KDIGO Stage 표시) |
| Creatinine 추이 | 실측값 + AI 예측선, KDIGO 임계값 기준선 |
| SHAP 피처 기여도 | 모델 결정에 영향을 준 주요 변수 시각화 |
| 권고 및 알림 | 위험도에 따른 약물 상호작용 경고 / 중복 처방 알림 |
| 임상 가이드라인 | 패혈증 관리 프로토콜, 처방 버튼 |
| 고위험 환자 목록 | 전체 환자 위험도 순 정렬 테이블 |

### 원무
- 신규 환자 정보 입력 (기본 정보 / 보호자 / 내원 사유 / 보험)
- 등록 진행 단계 표시 (정보 입력 → 서류 제출 → 최종 확인 → 등록 완료)
- 빠른 등록 기능, 업무 알림

### 마이페이지
- 사번, 이름, 소속 부서, 비밀번호 확인·수정

---

## 기술 스택

| 구분 | 기술 |
|---|---|
| 프레임워크 | Streamlit ≥ 1.35 |
| 데이터 처리 | Pandas, NumPy |
| 시각화 | Plotly |
| 예측 모델 | XGBoost (더미 데이터 적용) |
| 설명 가능 AI | SHAP |
| 기준 | KDIGO 2012 AKI 진단 기준 |

---

## 설치 및 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 앱 실행
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

---

## AKI 예측 시간대

```
현재  ──►  6h  ──►  12h  ──►  24h  ──►  48h
           저위험(<40%)  중위험(40~70%)  고위험(≥70%)
```

- **저위험 (< 40%)** — 정기 모니터링 유지
- **중위험 (40–70%)** — 4~6시간 내 재평가 권장
- **고위험 (≥ 70%)** — 즉각적인 신장 보호 중재 권장

---

## 폴더 구조

```
aki_cdss/
├── app.py            # 메인 Streamlit 앱 (멀티 페이지)
├── requirements.txt  # 의존성 목록
└── README.md
```

## 향후 연결 예정

| 위치 | 교체 내용 |
|---|---|
| `generate_patients()` | MIMIC-IV SQL 파이프라인 결과 DataFrame |
| `generate_shap()` | 실제 XGBoost + SHAP 계산값 |
| `generate_cr_series()` | 실제 환자 chartevents 시계열 |
| `AKI_6h / 12h / 24h / 48h` | XGBoost `predict_proba()` 결과 |

---

## 참고 문헌

- KDIGO Clinical Practice Guideline for Acute Kidney Injury, 2012
- SCCM Sepsis Management Guidelines, 2021
- Johnson, A. et al. MIMIC-IV Clinical Database, PhysioNet

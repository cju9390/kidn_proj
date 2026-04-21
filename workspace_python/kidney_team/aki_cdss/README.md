# AKI-CDSS | Streamlit 프로토타입

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 http://localhost:8501 접속

## 구조

```
aki_cdss/
├── app.py              # 메인 앱 (현재 더미 데이터)
├── requirements.txt
└── README.md
```

## 나중에 연결할 것들

| 위치 | 교체 내용 |
|------|-----------|
| `generate_patients()` | MIMIC-IV SQL 파이프라인 결과 DataFrame |
| `generate_shap_values()` | 실제 XGBoost + SHAP 계산값 |
| `generate_cr_trend()` | 실제 환자 chartevents 시계열 |
| `risk_score` 컬럼 | XGBoost predict_proba() 결과 |

## 탭 구성

- **Tab 1**: AKI 조기 경보 — Risk 게이지, SHAP 바 차트, Cr 트렌드, 고위험 환자 테이블
- **Tab 3**: 환자 대시보드 — 요약 카드, Cr/UO 시계열, 레이더 차트, 임상 노트

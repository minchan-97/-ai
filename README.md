[README.md](https://github.com/user-attachments/files/28634030/README.md)
# Logic AI Streamlit API Demo

실시간 불일치율 기반 신뢰도 검증 시스템 Streamlit 버전입니다.

## 실행

```bash
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## OpenAI 공식 URL

```text
https://api.openai.com/v1/chat/completions
```

## 모드

- `Mock`: API 키 없이 테스트
- `OpenAI-compatible`: OpenAI / vLLM / LM Studio / OpenRouter 등 chat-completions 호환 API
- `Custom JSON API`: `{prompt, system_prompt, model, temperature, max_tokens}`를 POST하고 `output`, `answer`, `text`, `content`, `response` 중 하나를 응답 필드로 받는 커스텀 API

## 동작

1. 같은 질문으로 자유 응답과 제어 응답을 각각 호출합니다.
2. 두 응답을 **OpenAI 임베딩**(`text-embedding-3-small` 기본)으로 벡터화합니다.
3. 두 임베딩의 cosine distance를 0~100으로 정규화해 불일치율로 씁니다(0=동일, 100=정반대).
4. mismatch가 threshold 이하이면 자유 응답을 채택합니다.
5. mismatch가 threshold 초과이면 제어 응답을 최종 답변으로 채택합니다.
6. 사용자 피드백을 SQLite DB(`logic_ai_data.db`)에 저장합니다.
7. Mock 모드 또는 임베딩 키가 없으면 글자 분포 폴백으로 동작하며,
   이 경우는 의미 반전을 감지하지 못하므로 결과 패널에 그 사실이 표시됩니다.

## 한계

- 임계값은 여전히 휴리스틱입니다. 고위험 도메인의 단독 판정 근거로 쓰면 안 됩니다.
- 두 응답이 비슷하다고 정답이라는 뜻은 아닙니다(둘 다 같이 틀릴 수 있음).
- 임베딩이 의미를 보긴 하지만 동의어/반의어 구분 능력에는 한계가 있으므로
  중요한 결정은 반드시 사람이 함께 검토하세요.

## 🧪 클라우드 모드 (실험적)

사이드바 하단에서 토글 가능. 기본 모드(2회 호출)와 별개로 동작합니다.

**원리**: 같은 system prompt로 LLM을 N번(기본 5번) 호출 → N개 임베딩의
점 클라우드 → 클라우드 위상/분산 특성으로 일관성 측정.

**측정 특성** (`cloud_features.py`):
- `basic_mean/max/std`: pairwise distance 통계
- `cluster_separation`: 2개 군집 분리도 (양분된 응답이면 큼)
- `cluster_balance`: 군집 크기 균형
- `spread_pc1_ratio / pc2_ratio`: 분산이 어느 축에 집중되는가
- `spread_effective_dim`: 응답들의 유효 차원
- `topo_h0_*`: persistent homology (gudhi 설치 시에만)

**가설**: 두 점 거리(기본 모드)만으로는 못 잡는 신호 — 예를 들어
"N개 응답이 두 군집으로 갈리는 패턴" — 을 클라우드 위상이 잡는다.

**한계 (정직하게 짚습니다)**:
- 비용이 N/2배 증가 (N=5면 호출 2.5배).
- temperature가 너무 낮으면 N개 응답이 같아져 신호가 사라짐 (0.7+ 권장).
- 단일 mismatch 환원은 `basic_mean`만 사용. 학습된 융합은 별도 평가 도구.
- **시뮬레이션 실험에선 가설이 살아남았지만, 진짜 LLM N회 호출 데이터로는
  아직 AUROC 검증이 안 됨.** 그래서 'EXPERIMENTAL' 라벨을 유지합니다.

**진짜 검증을 하려면**: 별도 평가 패키지(`logic_ai_cloud_eval`)에서
실제 LLM N회 호출 응답으로 AUROC를 측정하세요.

## 🚨 안전 누적 차단 (v4 추가)

사이드바 하단에서 토글 가능. **기본 OFF**.

**원리**: 최근 N회 호출의 판정을 누적 모니터링. INCONSISTENT 비율이
임계값(기본 50%)을 넘으면 시스템을 자동 잠금. 관리자 다중 승인(기본 2명)이
있어야 해제.

**의도된 사용 시나리오**:
- 의료 챗봇이 환각 의심 응답을 연속 출력할 때 자동 정지
- 교사용 글 검토 도구에서 다수의 의심 결과 누적 시 검토 일시 중단
- 운영 중인 LLM 서비스에 일종의 circuit breaker 추가

**조정 가능한 파라미터**:
- `window_size`: 최근 N회 추적 (기본 10)
- `inconsistent_threshold`: 잠금 트리거 비율 (기본 0.5)
- `min_samples`: 최소 표본 수 (기본 5, 작은 표본 보호)
- `required_signatures`: 해제 필요 승인 수 (기본 2)
- `authorized_signers`: 승인 가능 ID 목록

**정직한 한계**:
- 모든 임계값은 휴리스틱. 도메인에 맞게 보정 필요.
- "관리자 ID"는 현재 클라이언트 측 텍스트 입력일 뿐 (진짜 인증 X).
  실제 배포 시에는 OAuth/SSO와 연결해야 함.
- "강제 초기화" 버튼은 응급 상황용. 운영 환경에서는 권한 분리 필요.

**Grav Prison 패키지에서 추출**: 원본은 물리/감옥 비유와 별도 모니터
스레드, 다중 백엔드를 가졌지만, 비유와 부수 기능을 다 버리고
"누적 감시 + 자동 차단 + 다중 승인 해제"라는 핵심 한 조각만 추출.

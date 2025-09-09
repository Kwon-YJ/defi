Stablecoin Support (USDC, USDT, DAI)

구현 개요
- Maker PSM: USDC↔DAI 근사 1:1 스왑 엣지(`maker_psm`)를 생성. PSM 주소가 설정되면 on-chain tin/tout을 반영.
- Curve StableSwap: Curve 레지스트리에서 페어가 존재하면 `curve.stableswap` 액션이 풀 상태 기반 엣지를 생성.
- Stable Aggregator Fallback: 집계 라우팅 근사 경로를 `stable_agg`로 추가(USDC↔USDT, USDC↔DAI, USDT↔DAI). 수수료 0.05% 가정, 유동성 가상치.
- Decimals/정규화: 모든 스왑은 `get_decimals`와 정규화 로직을 통해 단위 차이를 보정.

환경변수
- `MAKER_PSM_USDC`: Maker PSM(USDC/DAI) 컨트랙트 주소를 지정하면 tin/tout을 사용(없으면 보수적 기본 수수료 사용).

주의사항
- Fallback 경로(`stable_agg`)는 근사값으로, 프루닝 시 다른 더 낮은 수수료/정확 경로가 있으면 지배되어 제거될 수 있습니다.
- 실시간 업데이트는 V2/V3/Curve 이벤트 기반으로 반영되며, PSM은 주기적 재빌드에서 수수료를 갱신합니다.


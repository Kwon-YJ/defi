Synthetix Synthetic Assets (sUSD, sETH)

구현 개요
- 액션: `synthetix.exchange`, `synthetix.mint`, `synthetix.burn`
- 주소: 메인넷 sUSD/sETH 프록시 주소를 사용(sUSD 0x57ab1e…, sETH 0x5e74c9…)
- 교환: Uniswap V2 WETH/USDC 및 WBTC/USDC 스팟을 이용해 sUSD per sETH/sBTC를 근사.
- 수수료: SystemSettings(exchangeFeeRate)가 구성되어 있으면 on-chain 값을 사용, 없으면 보수적 기본(0.3%).
- 발행/소각: SNX 담보를 기반으로 sUSD를 mint/burn하는 근사 경로 제공(보수적 비율/안전계수 적용).

토큰 포함 옵션
- `INCLUDE_SYNTH_TOKENS=1`로 sUSD/sETH를 토큰 셋에 자동 포함(그래프 연결성 향상).

주의사항
- 실제 Synthetix 환전/발행 수수료는 자산별로 다를 수 있으며, 본 구현은 안전한 상한선 근사를 적용합니다.
- 오프체인 가격 근사를 사용할 때는 프루닝에서 정확도 높은 경로(예: stables/Curve)가 우선되도록 설계했습니다.


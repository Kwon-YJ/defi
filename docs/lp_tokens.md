LP Tokens Support

개요
- Uniswap V2 LP 토큰(UNI-V2)과 Curve LP 토큰을 그래프에 합리적 근사로 모델링합니다.

Uniswap V2 (UNI-V2)
- 액션: `uniswap_v2_lp_add`, `uniswap_v2_lp_remove`
- 원리: 풀 리저브(r0,r1), totalSupply, decimals를 활용해
  - add: 1 토큰 입금 시 발행 LP 수량 L = ts/r (proportional)
  - remove: 1 LP 소각 시 수령 토큰 수량 = r/ts
- 식별자: `lp:v2:<pair_address>`

Curve LP
- 액션: `curve_lp_add`, `curve_lp_remove`
- 원리: 풀의 `calc_token_amount`, `calc_withdraw_one_coin` 호출로 정확 근사
- LP 정보: registry/풀 인터페이스에서 LP 토큰 주소/decimals/totalSupply 조회 후 메타데이터로 기록
- 식별자: `lp:curve:<pool_address>`

메타데이터
- 공통: `lp_token`, `lp_decimals`, `lp_total_supply` 등 기록
- Curve는 풀 파라미터(A, fee/admin_fee)와 코인 인덱스 등 추가 기록

주의사항
- 일부 Curve 풀은 stableswap이 아니므로 단순 수수료/공식이 다를 수 있음(예: tricrypto). 본 구현은 공용 ABI(get_dy 등)가 제공되는 경우 보수적으로 동작합니다.


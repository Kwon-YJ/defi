논문(2103.02228) 기준 25개 자산 목록 및 주소 매핑 가이드

개요
- 원문에는 ETH를 기준으로 24개 토큰과의 거래 쌍이 제시되어 총 25개 자산으로 구성됩니다.
- 본 저장소에서는 온체인 상호작용을 위해 ETH를 WETH(랩드 이더) 주소로 사용합니다.
- 주소/decimals는 `config/paper_assets_25.json`에서 관리됩니다.

자산 목록(심볼)
- ETH (on-chain: WETH 사용)
- AMN, AMPL, ANT, BAT, BNT, DAI, DATA, ENJ, FXC, GNO, HEDG, KNC, MANA, MKR, POA20, RCN, RDN, RLC, SAI, SAN, SNT, TKN, TRST, UBT

구성 방법
- `config/paper_assets_25.json` 파일을 열어 각 심볼의 `address`와 `decimals`를 채워주세요.
- 본 저장소에는 WETH/DAI는 기본 제공되며, 나머지는 정확한 메인넷 주소를 입력해야 합니다.
- 민감한 변경은 PR 또는 커밋 메시지에 근거(공식 컨트랙트/프로젝트 링크)를 남겨주세요.

실행 설정
- 환경변수 `USE_PAPER_25_ASSETS=1`로 활성화합니다.
- 주소가 설정되지 않은 심볼은 자동으로 제외되며, 로그 경고가 출력됩니다.

주의사항
- 일부 토큰은 리브랜딩/마이그레이션 기록이 있으므로 당시(2019~2020) 사용 컨트랙트를 정확히 사용해야 합니다.
- SAI(싱글 담보 DAI) 등 레거시 토큰은 상호운용 로직에서 보수적으로 취급하세요.


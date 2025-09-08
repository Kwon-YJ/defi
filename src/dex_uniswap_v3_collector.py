import json
from typing import Dict, List, Optional, Tuple
from web3 import Web3
from src.logger import setup_logger

logger = setup_logger(__name__)


class UniswapV3Collector:
    """Uniswap V3 데이터 수집기 (가격/유동성 기본 반영)

    - 최소 ABI만 사용하여 팩토리 및 풀의 핵심 상태(slot0, liquidity, fee, token0/1)를 조회
    - 복잡한 집중 유동성 구조는 단순화하여 그래프에 환율 및 대략적 유동성으로 반영
    """

    FACTORY_ADDRESS = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
    FEE_TIERS = [500, 3000, 10000]  # 0.05%, 0.3%, 1%
    FEE_LABELS = {500: "0.05%", 3000: "0.3%", 10000: "1%"}

    def __init__(self, web3_provider: Web3):
        self.w3 = web3_provider
        self.factory_abi = self._get_factory_abi()
        self.pool_abi = self._get_pool_abi()
        self._last_fee_inside: Dict[str, Dict[Tuple[int, int], Tuple[int, int]]] = {}
        try:
            with open('abi/erc20.json', 'r') as f:
                self.erc20_abi = json.load(f)
        except FileNotFoundError:
            self.erc20_abi = [
                {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
                {"constant": True, "inputs": [], "name": "symbol",   "outputs": [{"name": "", "type": "string"}], "type": "function"},
                {"constant": True, "inputs": [], "name": "name",     "outputs": [{"name": "", "type": "string"}], "type": "function"}
            ]
        self.factory = self.w3.eth.contract(address=self.FACTORY_ADDRESS, abi=self.factory_abi)

    def _get_factory_abi(self) -> List[Dict]:
        return [
            {
                "inputs": [
                    {"internalType": "address", "name": "tokenA", "type": "address"},
                    {"internalType": "address", "name": "tokenB", "type": "address"},
                    {"internalType": "uint24",  "name": "fee",    "type": "uint24"}
                ],
                "name": "getPool",
                "outputs": [{"internalType": "address", "name": "pool", "type": "address"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]

    def _get_pool_abi(self) -> List[Dict]:
        return [
            {
                "inputs": [],
                "name": "slot0",
                "outputs": [
                    {"internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
                    {"internalType": "int24",   "name": "tick",         "type": "int24"},
                    {"internalType": "uint16",  "name": "observationIndex", "type": "uint16"},
                    {"internalType": "uint16",  "name": "observationCardinality", "type": "uint16"},
                    {"internalType": "uint16",  "name": "observationCardinalityNext", "type": "uint16"},
                    {"internalType": "uint8",   "name": "feeProtocol", "type": "uint8"},
                    {"internalType": "bool",    "name": "unlocked",    "type": "bool"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {"inputs": [], "name": "liquidity", "outputs": [{"internalType": "uint128", "name": "", "type": "uint128"}], "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "token0",   "outputs": [{"internalType": "address", "name": "", "type": "address"}], "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "token1",   "outputs": [{"internalType": "address", "name": "", "type": "address"}], "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "fee",      "outputs": [{"internalType": "uint24",  "name": "", "type": "uint24"}],  "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "tickSpacing", "outputs": [{"internalType": "int24", "name": "", "type": "int24"}], "stateMutability": "view", "type": "function"}
            ,
            {"inputs": [], "name": "feeGrowthGlobal0X128", "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "feeGrowthGlobal1X128", "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
            {
                "inputs": [{"internalType": "int24", "name": "tick", "type": "int24"}],
                "name": "ticks",
                "outputs": [
                    {"internalType": "uint128", "name": "liquidityGross", "type": "uint128"},
                    {"internalType": "int128",  "name": "liquidityNet",   "type": "int128"},
                    {"internalType": "uint256", "name": "feeGrowthOutside0X128", "type": "uint256"},
                    {"internalType": "uint256", "name": "feeGrowthOutside1X128", "type": "uint256"},
                    {"internalType": "int56",   "name": "tickCumulativeOutside", "type": "int56"},
                    {"internalType": "uint160",  "name": "secondsPerLiquidityOutsideX128", "type": "uint160"},
                    {"internalType": "uint32",   "name": "secondsOutside", "type": "uint32"},
                    {"internalType": "bool",     "name": "initialized", "type": "bool"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]

    async def get_pool_address(self, token0: str, token1: str, fee: int) -> Optional[str]:
        try:
            pool = self.factory.functions.getPool(token0, token1, fee).call()
            if pool == "0x0000000000000000000000000000000000000000":
                return None
            return pool
        except Exception as e:
            logger.debug(f"UniswapV3 getPool 실패 {token0[:6]}-{token1[:6]} fee={fee}: {e}")
            return None

    async def get_pool_core_state(self, pool_address: str) -> Optional[Dict]:
        try:
            pool = self.w3.eth.contract(address=pool_address, abi=self.pool_abi)
            slot0 = pool.functions.slot0().call()
            sqrt_price_x96 = slot0[0]
            current_tick = int(slot0[1])
            liq = pool.functions.liquidity().call()
            t0 = pool.functions.token0().call()
            t1 = pool.functions.token1().call()
            fee = pool.functions.fee().call()
            tick_spacing = int(pool.functions.tickSpacing().call())

            dec0 = self._get_decimals(t0)
            dec1 = self._get_decimals(t1)

            return {
                'sqrtPriceX96': int(sqrt_price_x96),
                'tick': current_tick,
                'liquidity': int(liq),
                'token0': t0,
                'token1': t1,
                'fee': int(fee),
                'tickSpacing': tick_spacing,
                'dec0': dec0,
                'dec1': dec1,
            }
        except Exception as e:
            logger.debug(f"UniswapV3 풀 상태 조회 실패 {pool_address[:6]}: {e}")
            return None

    def _get_decimals(self, token_addr: str) -> int:
        try:
            c = self.w3.eth.contract(address=token_addr, abi=self.erc20_abi)
            return int(c.functions.decimals().call())
        except Exception:
            return 18

    @staticmethod
    def price_from_sqrtX96(sqrt_price_x96: int, dec0: int, dec1: int) -> float:
        if sqrt_price_x96 <= 0:
            return 0.0
        ratio_x128 = (sqrt_price_x96 * sqrt_price_x96)
        price = ratio_x128 / float(2 ** 192)
        price *= 10 ** (dec0 - dec1)
        return float(price)

    @staticmethod
    def sqrt_from_x96_normalized(sqrt_price_x96: int, dec0: int, dec1: int) -> float:
        """정규화된 sqrtP 계산 (decimals 차 반영).

        sqrtP_norm = (sqrtPriceX96 / 2**96) * 10**((dec0 - dec1)/2)
        """
        if sqrt_price_x96 <= 0:
            return 0.0
        q96 = float(2 ** 96)
        dec_adj = 10 ** ((dec0 - dec1) / 2.0)
        return float(sqrt_price_x96) / q96 * float(dec_adj)

    @staticmethod
    def sqrt_from_tick_normalized(tick: int, dec0: int, dec1: int) -> float:
        """Tick로부터 정규화된 sqrt(price) 계산 (decimals 차 반영)."""
        sqrt_raw = 1.0001 ** (tick / 2.0)
        dec_adj = 10 ** ((dec0 - dec1) / 2.0)
        return float(sqrt_raw) * float(dec_adj)

    @staticmethod
    def liquidity_per_token0(sqrtP: float, sqrtA: float, sqrtB: float) -> float:
        """현재 sqrtP, 범위 [sqrtA, sqrtB]에서 1 token0 당 민트되는 유동성 L."""
        try:
            if sqrtP <= 0 or sqrtA <= 0 or sqrtB <= 0 or sqrtA >= sqrtB:
                return 0.0
            if sqrtP <= sqrtA:
                return (sqrtA * sqrtB) / (sqrtB - sqrtA)
            if sqrtP < sqrtB:
                return (sqrtP * sqrtB) / (sqrtB - sqrtP)
            return 0.0
        except Exception:
            return 0.0

    @staticmethod
    def liquidity_per_token1(sqrtP: float, sqrtA: float, sqrtB: float) -> float:
        """현재 sqrtP, 범위 [sqrtA, sqrtB]에서 1 token1 당 민트되는 유동성 L."""
        try:
            if sqrtP <= 0 or sqrtA <= 0 or sqrtB <= 0 or sqrtA >= sqrtB:
                return 0.0
            if sqrtP <= sqrtA:
                return 0.0
            if sqrtP < sqrtB:
                return 1.0 / (sqrtP - sqrtA)
            return 1.0 / (sqrtB - sqrtA)
        except Exception:
            return 0.0

    @staticmethod
    def amounts_per_liquidity(sqrtP: float, sqrtA: float, sqrtB: float) -> Tuple[float, float]:
        """유동성 L=1 소각 시 반환되는 (amount0, amount1) (정규화 단위)."""
        try:
            if sqrtP <= 0 or sqrtA <= 0 or sqrtB <= 0 or sqrtA >= sqrtB:
                return 0.0, 0.0
            if sqrtP <= sqrtA:
                # 전량 token0
                return (sqrtB - sqrtA) / (sqrtA * sqrtB), 0.0
            if sqrtP < sqrtB:
                # 양 토큰 혼합
                amt0 = (sqrtB - sqrtP) / (sqrtP * sqrtB)
                amt1 = (sqrtP - sqrtA)
                return max(amt0, 0.0), max(amt1, 0.0)
            # 전량 token1
            return 0.0, (sqrtB - sqrtA)
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def price_from_tick(tick: int, dec0: int, dec1: int) -> float:
        # Uniswap V3 tick price: 1.0001 ** tick, adjust for decimals
        p = (1.0001 ** float(tick)) * (10 ** (dec0 - dec1))
        return float(p)

    @staticmethod
    def estimate_pseudo_reserves(price01: float, base_liquidity: float = 100.0) -> Tuple[float, float]:
        r0 = max(1e-9, float(base_liquidity))
        r1 = max(1e-9, float(base_liquidity) * max(price01, 0.0))
        return (r0, r1)

    # --- Fee growth helpers ---
    @staticmethod
    def _u256(x: int) -> int:
        return x & ((1 << 256) - 1)

    @classmethod
    def _u256_sub(cls, a: int, b: int) -> int:
        return cls._u256(a - b)

    async def _fee_growth_inside(self, pool_address: str, cur_tick: int, tick_lower: int, tick_upper: int) -> Tuple[int, int]:
        pool = self.w3.eth.contract(address=pool_address, abi=self.pool_abi)
        fgg0 = int(pool.functions.feeGrowthGlobal0X128().call())
        fgg1 = int(pool.functions.feeGrowthGlobal1X128().call())
        tL = pool.functions.ticks(int(tick_lower)).call()
        tU = pool.functions.ticks(int(tick_upper)).call()
        out0L = int(tL[2]); out1L = int(tL[3])
        out0U = int(tU[2]); out1U = int(tU[3])
        # below
        if cur_tick >= tick_lower:
            below0 = out0L
            below1 = out1L
        else:
            below0 = self._u256_sub(fgg0, out0L)
            below1 = self._u256_sub(fgg1, out1L)
        # above
        if cur_tick < tick_upper:
            above0 = self._u256_sub(fgg0, out0U)
            above1 = self._u256_sub(fgg1, out1U)
        else:
            above0 = out0U
            above1 = out1U
        # inside = global - below - above
        inside0 = self._u256_sub(self._u256_sub(fgg0, below0), above0)
        inside1 = self._u256_sub(self._u256_sub(fgg1, below1), above1)
        return inside0, inside1

    async def estimate_fees_per_L(self, pool_address: str, dec0: int, dec1: int,
                                  tick_lower: int, tick_upper: int, cur_tick: int) -> Tuple[float, float]:
        """해당 풀과 밴드에서 직전 관측 이후 1 L 당 발생한 수수료(정규화 단위)를 추정.

        반환값: (token0_per_L, token1_per_L)
        """
        inside0, inside1 = await self._fee_growth_inside(pool_address, cur_tick, tick_lower, tick_upper)
        last_map = self._last_fee_inside.setdefault(pool_address, {})
        last0, last1 = last_map.get((tick_lower, tick_upper), (inside0, inside1))
        # delta with wraparound
        d0 = self._u256_sub(inside0, last0)
        d1 = self._u256_sub(inside1, last1)
        # store current
        last_map[(tick_lower, tick_upper)] = (inside0, inside1)
        Q128 = float(2 ** 128)
        # normalize by decimals
        amt0 = (float(d0) / Q128) / (10 ** dec0)
        amt1 = (float(d1) / Q128) / (10 ** dec1)
        # keep non-negative
        return max(0.0, amt0), max(0.0, amt1)

    async def build_edges_for_pair(self, tokenA: str, tokenB: str) -> List[Dict]:
        edges: List[Dict] = []
        for fee in self.FEE_TIERS:
            pool = await self.get_pool_address(tokenA, tokenB, fee)
            if not pool:
                continue
            state = await self.get_pool_core_state(pool)
            if not state:
                continue
            price01 = self.price_from_sqrtX96(state['sqrtPriceX96'], state['dec0'], state['dec1'])
            if price01 <= 0:
                continue
            # Tick-range 모델링: 현재 tick을 기준으로 한 밴드를 정의 (한 칸 폭)
            ts = int(state.get('tickSpacing', 60) or 60)
            cur_tick = int(state.get('tick', 0) or 0)
            tick_lower = (cur_tick // ts) * ts
            tick_upper = tick_lower + ts
            reserve0, reserve1 = self.estimate_pseudo_reserves(price01, base_liquidity=100.0)
            edges.append({
                'token0': state['token0'],
                'token1': state['token1'],
                'pool': pool,
                'fee_fraction': state['fee'] / 1_000_000.0,
                'fee_tier': int(state['fee']),
                'fee_label': self.FEE_LABELS.get(int(state['fee']), str(int(state['fee']))),
                'reserve0': reserve0,
                'reserve1': reserve1,
                'tick': cur_tick,
                'tick_lower': int(tick_lower),
                'tick_upper': int(tick_upper),
                'tick_spacing': ts,
                'sqrtPriceX96': int(state['sqrtPriceX96']),
            })
        return edges

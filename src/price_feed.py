from typing import Dict, Optional, List, Tuple
from web3 import Web3
from src.logger import setup_logger
from src.data_storage import DataStorage

logger = setup_logger(__name__)


class PriceFeed:
    """간단한 온체인 스팟 기반 실시간 가격 피드 (Uniswap V2 기반 근사).

    - WETH/USDC 풀이 존재한다고 가정하고 WETH USD 가격을 계산.
    - 각 토큰은 WETH 또는 USDC와의 페어를 통해 USD 가격을 근사.
    - 결과는 Redis(DataStorage)에 TTL과 함께 저장.
    """

    V2_FACTORY = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"

    def __init__(self, w3: Web3, storage: Optional[DataStorage] = None):
        self.w3 = w3
        self.storage = storage or DataStorage()
        self.factory = self.w3.eth.contract(address=self.V2_FACTORY, abi=[{
            "constant": True,
            "inputs": [{"name": "tokenA", "type": "address"}, {"name": "tokenB", "type": "address"}],
            "name": "getPair", "outputs": [{"name": "pair", "type": "address"}], "type": "function"
        }])
        self.pair_abi = [
            {"constant": True, "inputs": [], "name": "getReserves", "outputs": [
                {"name": "reserve0", "type": "uint112"},
                {"name": "reserve1", "type": "uint112"},
                {"name": "blockTimestampLast", "type": "uint32"}
            ], "type": "function"},
            {"constant": True, "inputs": [], "name": "token0", "outputs": [{"name": "", "type": "address"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "token1", "outputs": [{"name": "", "type": "address"}], "type": "function"}
        ]

    def _pair(self, a: str, b: str) -> Optional[str]:
        try:
            p = self.factory.functions.getPair(a, b).call()
            if p and int(p, 16) != 0:
                return p
        except Exception as e:
            logger.debug(f"V2 getPair 실패 {a[:6]}-{b[:6]}: {e}")
        return None

    def _reserves(self, pair: str) -> Optional[Tuple[str, str, int, int]]:
        try:
            c = self.w3.eth.contract(address=pair, abi=self.pair_abi)
            t0 = c.functions.token0().call()
            t1 = c.functions.token1().call()
            r0, r1, _ = c.functions.getReserves().call()
            return t0, t1, int(r0), int(r1)
        except Exception as e:
            logger.debug(f"V2 getReserves 실패 {pair[:6]}: {e}")
            return None

    def _price_tokenB_per_tokenA(self, a: str, b: str, dec_a: int, dec_b: int) -> float:
        """1 a당 b의 수량 반환."""
        try:
            pr = self._pair(a, b)
            if not pr:
                return 0.0
            t0, t1, r0, r1 = self._reserves(pr) or (None, None, 0, 0)
            if not t0 or r0 == 0 or r1 == 0:
                return 0.0
            if t0.lower() == a.lower() and t1.lower() == b.lower():
                return (r1 / (10 ** dec_b)) / (r0 / (10 ** dec_a))
            elif t0.lower() == b.lower() and t1.lower() == a.lower():
                return (r0 / (10 ** dec_b)) / (r1 / (10 ** dec_a))
            return 0.0
        except Exception:
            return 0.0

    async def update_prices_for(self, tokens: Dict[str, str]) -> int:
        """토큰 주소 사전(sym->addr)에 대해 USD 가격을 업데이트. 성공 건수 반환."""
        try:
            usdc = tokens.get('USDC') or tokens.get('usdc')
            weth = tokens.get('WETH') or tokens.get('weth')
            if not (usdc and weth):
                return 0
            usdc_dec = 6
            weth_dec = 18
            usdc_per_weth = self._price_tokenB_per_tokenA(weth, usdc, weth_dec, usdc_dec)
            if usdc_per_weth <= 0:
                return 0
            updated = 0
            for sym, addr in tokens.items():
                try:
                    if addr.lower() == usdc.lower():
                        price = 1.0
                    elif sym.upper() in ('DAI', 'USDT', 'TUSD'):
                        price = 1.0
                    elif addr.lower() == weth.lower():
                        price = usdc_per_weth
                    else:
                        # Try token/WETH; if not, token/USDC
                        p1 = self._price_tokenB_per_tokenA(addr, weth, 18, 18)
                        if p1 > 0:
                            price = usdc_per_weth * p1
                        else:
                            p2 = self._price_tokenB_per_tokenA(addr, usdc, 18, usdc_dec)
                            price = p2 if p2 > 0 else 0.0
                    if price > 0:
                        await self.storage.store_token_price(addr, float(price))
                        updated += 1
                except Exception:
                    continue
            return updated
        except Exception as e:
            logger.debug(f"가격 업데이트 실패: {e}")
            return 0


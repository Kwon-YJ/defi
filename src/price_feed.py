from typing import Dict, Optional, List, Tuple
from web3 import Web3
from src.logger import setup_logger
from src.data_storage import DataStorage
from config.config import config

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
        # ERC20 decimals ABI (minimal)
        self.erc20_abi = [{"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"}]
        # caches
        self.token_pairs: Dict[str, Tuple[str, str]] = {}
        # token -> (pair_addr, base_token_label: 'WETH'|'USDC')
        self.token_to_pair: Dict[str, Tuple[str, str]] = {}
        self.WETH: Optional[str] = None
        self.USDC: Optional[str] = None

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
                        await self._validated_store(addr, float(price), is_stable=(sym.upper() in ('USDC','USDT','DAI','TUSD','SUSD')))
                        updated += 1
                except Exception:
                    continue
            return updated
        except Exception as e:
            logger.debug(f"가격 업데이트 실패: {e}")
            return 0

    def _decimals(self, addr: str, default: int = 18) -> int:
        try:
            c = self.w3.eth.contract(address=addr, abi=self.erc20_abi)
            return int(c.functions.decimals().call())
        except Exception:
            return default

    def build_pairs(self, tokens: Dict[str, str]) -> None:
        """가격 피드가 추적할 V2 pair 주소들을 미리 계산해 캐시한다."""
        try:
            self.USDC = tokens.get('USDC') or tokens.get('usdc')
            self.WETH = tokens.get('WETH') or tokens.get('weth')
            if not (self.USDC and self.WETH):
                return
            # Cache WETH/USDC
            wu = self._pair(self.WETH, self.USDC)
            if wu:
                self.token_pairs['WETH/USDC'] = (wu, 'spot')
            # For each token, prefer token/WETH then token/USDC
            for sym, addr in tokens.items():
                if sym.upper() in ('USDC', 'WETH'):
                    continue
                pair = self._pair(addr, self.WETH)
                base = 'WETH'
                if not pair:
                    pair = self._pair(addr, self.USDC)
                    base = 'USDC'
                if pair:
                    self.token_to_pair[addr.lower()] = (pair, base)
        except Exception:
            return

    async def handle_log(self, log: Dict) -> None:
        """Swap/Sync 이벤트를 받아 관련 토큰 USD 가격을 갱신한다."""
        try:
            addr = log.get('address')
            if not isinstance(addr, str):
                return
            topics = log.get('topics') or []
            if not topics:
                return
            # only react to Swap/Sync
            t0 = topics[0]
            # hashed topics from RealTimeDataCollector: we'll accept any event if address matches a tracked pair
            addr_l = addr.lower()
            # WETH/USDC pair update → refresh WETH price and dependent tokens
            if any(addr_l == p[0].lower() for p in self.token_pairs.values()):
                # update WETH price
                usdc_dec = 6
                weth_dec = 18
                if not (self.WETH and self.USDC):
                    return
                usdc_per_weth = self._price_tokenB_per_tokenA(self.WETH, self.USDC, weth_dec, usdc_dec)
                if usdc_per_weth > 0:
                    await self._validated_store(self.WETH, float(usdc_per_weth), is_stable=False)
                    # refresh all tracked tokens against updated WETH price
                    for tok, (pair, base) in list(self.token_to_pair.items()):
                        try:
                            if base == 'WETH':
                                p = self._price_tokenB_per_tokenA(tok, self.WETH, self._decimals(tok), weth_dec)
                                price = p * usdc_per_weth if p > 0 else 0.0
                            else:
                                price = self._price_tokenB_per_tokenA(tok, self.USDC, self._decimals(tok), usdc_dec)
                            if price > 0:
                                await self._validated_store(tok, float(price), is_stable=False)
                        except Exception:
                            continue
                return
            # If log comes from a token pair we track, update that token price only
            for tok, (pair, base) in list(self.token_to_pair.items()):
                if addr_l == pair.lower():
                    usdc_dec = 6
                    weth_dec = 18
                    if base == 'WETH':
                        # get current WETH price
                        p_w = await self.storage.get_token_price(self.WETH) if self.WETH else None
                        usdc_per_weth = float(p_w.get('price_usd')) if p_w else 0.0
                        if usdc_per_weth <= 0 and self.WETH and self.USDC:
                            usdc_per_weth = self._price_tokenB_per_tokenA(self.WETH, self.USDC, weth_dec, usdc_dec)
                        p = self._price_tokenB_per_tokenA(tok, self.WETH, self._decimals(tok), weth_dec)
                        price = p * usdc_per_weth if (p > 0 and usdc_per_weth > 0) else 0.0
                    else:
                        price = self._price_tokenB_per_tokenA(tok, self.USDC, self._decimals(tok), usdc_dec)
                    if price > 0:
                        await self._validated_store(tok, float(price), is_stable=False)
                    return
        except Exception:
            return

    async def _validated_store(self, token: str, price: float, *, is_stable: bool) -> None:
        """간단한 이상치 필터 및 EMA 스무딩 후 저장.

        - 급등락 캡: PRICE_JUMP_MAX_PCT (기본 20%)
        - 스테이블코인 앵커: PRICE_STABLE_MAX_DEV (기본 3%)
        - EMA 스무딩: PRICE_EMA_ALPHA (기본 0.2)
        """
        try:
            alpha = float(getattr(config, 'price_ema_alpha', 0.2))
            jump = float(getattr(config, 'price_jump_max_pct', 0.2))
            sdev = float(getattr(config, 'price_stable_max_dev', 0.03))
            prev = await self.storage.get_token_price(token)
            prev_p = float(prev.get('price_usd')) if prev and 'price_usd' in prev else None
            p = float(price)
            if is_stable:
                # clamp around $1
                lo = 1.0 - sdev
                hi = 1.0 + sdev
                if p < lo:
                    p = lo
                elif p > hi:
                    p = hi
            if prev_p and prev_p > 0:
                # cap sudden jumps
                max_up = prev_p * (1.0 + jump)
                max_dn = prev_p * (1.0 - jump)
                if p > max_up:
                    p = max_up
                elif p < max_dn:
                    p = max_dn
                # EMA smoothing
                p = alpha * p + (1.0 - alpha) * prev_p
            await self.storage.store_token_price(token, float(p))
        except Exception:
            # fallback to raw store
            try:
                await self.storage.store_token_price(token, float(price))
            except Exception:
                pass

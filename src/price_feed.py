from typing import Dict, Optional, List, Tuple
from web3 import Web3
from src.logger import setup_logger
from src.data_storage import DataStorage
from config.config import config
from src.rate_limit import RateLimiter
from src.dex_uniswap_v3_collector import UniswapV3Collector
from src.dex_curve_collector import CurveStableSwapCollector

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
        # V3 and Curve collectors
        self.v3 = UniswapV3Collector(self.w3)
        self.curve = CurveStableSwapCollector(self.w3)
        # rate limiter
        cps = float(getattr(config, 'price_calls_per_sec', 30.0)) if hasattr(config, 'price_calls_per_sec') else 30.0
        self.limiter = RateLimiter(calls_per_sec=cps)

    def _pair(self, a: str, b: str) -> Optional[str]:
        try:
            # rate limit
            import asyncio
            if asyncio.get_event_loop().is_running():
                # no-op if not awaited; it's okay for best effort
                pass
            p = self.factory.functions.getPair(a, b).call()
            if p and int(p, 16) != 0:
                return p
        except Exception as e:
            logger.debug(f"V2 getPair 실패 {a[:6]}-{b[:6]}: {e}")
        return None

    def _reserves(self, pair: str) -> Optional[Tuple[str, str, int, int]]:
        try:
            # rate limit best-effort (sync call; limiter mainly for async usage)
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
            raw_prices: Dict[str, float] = {}
            stable_syms = {'USDC','USDT','DAI','TUSD','SUSD'}
            for sym, addr in tokens.items():
                try:
                    if addr.lower() == usdc.lower():
                        price = 1.0
                    elif sym.upper() in ('DAI', 'USDT', 'TUSD'):
                        price = 1.0
                    elif addr.lower() == weth.lower():
                        price = usdc_per_weth
                    else:
                        price = self._agg_price_usd(addr, weth, usdc, usdc_per_weth, usdc_dec)
                    if price > 0:
                        raw_prices[addr] = float(price)
                except Exception:
                    continue
            await self._validated_store_bulk(raw_prices, stable_syms, tokens)
            return len(raw_prices)
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
                    # refresh all tracked tokens against updated WETH price (bulk)
                    price_map = {}
                    if self.WETH:
                        price_map[self.WETH] = float(usdc_per_weth)
                    for tok, (pair, base) in list(self.token_to_pair.items()):
                        try:
                            price = self._agg_price_usd(tok, self.WETH, self.USDC, usdc_per_weth, usdc_dec)
                            if price > 0:
                                price_map[tok] = float(price)
                        except Exception:
                            continue
                    await self._validated_store_bulk(price_map, set(), None)
                return
            # If log comes from a token pair we track, update that token price only
            for tok, (pair, base) in list(self.token_to_pair.items()):
                if addr_l == pair.lower():
                    usdc_dec = 6
                    weth_dec = 18
                    # Use aggregator regardless of base
                    p_w = await self.storage.get_token_price(self.WETH) if self.WETH else None
                    usdc_per_weth = float(p_w.get('price_usd')) if (p_w and p_w.get('price_usd')) else 0.0
                    if usdc_per_weth <= 0 and self.WETH and self.USDC:
                        usdc_per_weth = self._price_tokenB_per_tokenA(self.WETH, self.USDC, weth_dec, usdc_dec)
                    price = self._agg_price_usd(tok, self.WETH, self.USDC, usdc_per_weth, usdc_dec)
                    if price > 0:
                        await self._validated_store_bulk({tok: float(price)}, set(), None)
                    return
        except Exception:
            return

    def _agg_price_usd(self, token: str, weth: str, usdc: str, usdc_per_weth: float, usdc_dec: int) -> float:
        """V2/V3/Curve 후보 가격에서 중앙값을 선택하여 USD 가격을 반환."""
        candidates: List[float] = []
        dec_tok = self._decimals(token)
        # V2 via WETH
        try:
            p1 = self._price_tokenB_per_tokenA(token, weth, dec_tok, 18)
            if p1 > 0 and usdc_per_weth > 0:
                candidates.append(p1 * usdc_per_weth)
        except Exception:
            pass
        # V2 via USDC
        try:
            p2 = self._price_tokenB_per_tokenA(token, usdc, dec_tok, usdc_dec)
            if p2 > 0:
                candidates.append(p2)
        except Exception:
            pass
        # V3 direct: token/WETH and token/USDC across fee tiers
        try:
            for fee in getattr(self.v3, 'FEE_TIERS', [500, 3000, 10000]):
                p_tw = self.v3.price_tokenB_per_tokenA_v3(token, weth, int(fee))
                if p_tw > 0 and usdc_per_weth > 0:
                    candidates.append(p_tw * usdc_per_weth)
                p_tu = self.v3.price_tokenB_per_tokenA_v3(token, usdc, int(fee))
                if p_tu > 0:
                    candidates.append(p_tu)
        except Exception:
            pass
        # Curve direct
        try:
            found = self.curve.find_pool_for_pair(token, usdc)
            if found:
                pool, i, j = found
                pc = self.curve.get_price(pool, i, j, token, usdc)
                if pc > 0:
                    candidates.append(pc)
        except Exception:
            pass
        if not candidates:
            return 0.0
        candidates.sort()
        mid = len(candidates) // 2
        if len(candidates) % 2 == 1:
            return candidates[mid]
        return (candidates[mid - 1] + candidates[mid]) / 2.0

    # Historical backfill support
    async def update_prices_for_at_block(self, tokens: Dict[str, str], block_number: int) -> int:
        """특정 블록에서의 가격을 계산/저장 (V2 스팟 기준, 보조적으로 Curve)."""
        try:
            usdc = tokens.get('USDC') or tokens.get('usdc')
            weth = tokens.get('WETH') or tokens.get('weth')
            if not (usdc and weth):
                return 0
            usdc_dec = 6
            weth_dec = 18
            # WETH price at block
            wu = self._pair(weth, usdc)
            if not wu:
                return 0
            pr = self.w3.eth.contract(address=wu, abi=self.pair_abi)
            try:
                t0 = pr.functions.token0().call(block_identifier=block_number)
                t1 = pr.functions.token1().call(block_identifier=block_number)
                r0, r1, _ = pr.functions.getReserves().call(block_identifier=block_number)
                if t0.lower() == weth.lower() and t1.lower() == usdc.lower():
                    usdc_per_weth = (r1 / 10**usdc_dec) / (r0 / 10**weth_dec)
                else:
                    usdc_per_weth = (r0 / 10**usdc_dec) / (r1 / 10**weth_dec)
            except Exception:
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
                        # V2 spot at block via WETH then USD
                        p1 = 0.0
                        # token/weth
                        pair = self._pair(addr, weth)
                        if pair:
                            c = self.w3.eth.contract(address=pair, abi=self.pair_abi)
                            try:
                                t0 = c.functions.token0().call(block_identifier=block_number)
                                t1 = c.functions.token1().call(block_identifier=block_number)
                                r0, r1, _ = c.functions.getReserves().call(block_identifier=block_number)
                                dec_tok = self._decimals(addr)
                                if t0.lower() == addr.lower() and t1.lower() == weth.lower():
                                    p1 = (r1 / 10**weth_dec) / (r0 / 10**dec_tok)
                                elif t0.lower() == weth.lower() and t1.lower() == addr.lower():
                                    p1 = (r0 / 10**weth_dec) / (r1 / 10**dec_tok)
                            except Exception:
                                p1 = 0.0
                        if p1 > 0:
                            price = p1 * usdc_per_weth
                        else:
                            # fallback token/usdc at block
                            pair = self._pair(addr, usdc)
                            if pair:
                                c = self.w3.eth.contract(address=pair, abi=self.pair_abi)
                                try:
                                    t0 = c.functions.token0().call(block_identifier=block_number)
                                    t1 = c.functions.token1().call(block_identifier=block_number)
                                    r0, r1, _ = c.functions.getReserves().call(block_identifier=block_number)
                                    dec_tok = self._decimals(addr)
                                    if t0.lower() == addr.lower() and t1.lower() == usdc.lower():
                                        price = (r1 / 10**usdc_dec) / (r0 / 10**dec_tok)
                                    elif t0.lower() == usdc.lower() and t1.lower() == addr.lower():
                                        price = (r0 / 10**usdc_dec) / (r1 / 10**dec_tok)
                                    else:
                                        price = 0.0
                                except Exception:
                                    price = 0.0
                            else:
                                price = 0.0
                    if price > 0:
                        await self._validated_store(addr, float(price), is_stable=(sym.upper() in ('USDC','USDT','DAI','TUSD','SUSD')))
                        updated += 1
                except Exception:
                    continue
            return updated
        except Exception:
            return 0

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

    async def _validated_store_bulk(self, price_map: Dict[str, float], stable_syms: set, tokens: Optional[Dict[str,str]]) -> None:
        """여러 토큰 가격을 한 번에 검증/스무딩/저장."""
        if not price_map:
            return
        try:
            alpha = float(getattr(config, 'price_ema_alpha', 0.2))
            jump = float(getattr(config, 'price_jump_max_pct', 0.2))
            sdev = float(getattr(config, 'price_stable_max_dev', 0.03))
            addrs = list(price_map.keys())
            prevs = await self.storage.get_token_prices_bulk(addrs)
            final: Dict[str, float] = {}
            for addr, price in price_map.items():
                prev = prevs.get(addr) if prevs else None
                prev_p = float(prev.get('price_usd')) if prev and 'price_usd' in prev else None
                p = float(price)
                # if tokens mapping is provided, determine stability by symbol
                is_stable = False
                if tokens:
                    sym = None
                    for k, v in tokens.items():
                        if v.lower() == addr.lower():
                            sym = k
                            break
                    if sym and sym.upper() in stable_syms:
                        is_stable = True
                # apply stable clamp
                if is_stable:
                    lo = 1.0 - sdev
                    hi = 1.0 + sdev
                    if p < lo:
                        p = lo
                    elif p > hi:
                        p = hi
                if prev_p and prev_p > 0:
                    max_up = prev_p * (1.0 + jump)
                    max_dn = prev_p * (1.0 - jump)
                    if p > max_up:
                        p = max_up
                    elif p < max_dn:
                        p = max_dn
                    p = alpha * p + (1.0 - alpha) * prev_p
                final[addr] = float(p)
            await self.storage.store_token_prices_bulk(final, with_history=True)
        except Exception:
            # fallback to per-item
            for addr, price in price_map.items():
                try:
                    await self.storage.store_token_price(addr, float(price))
                except Exception:
                    pass

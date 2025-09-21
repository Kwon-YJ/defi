from typing import List, Tuple, Optional, Dict
from web3 import Web3
from src.logger import setup_logger

logger = setup_logger(__name__)


class MultiCaller:
    """Minimal Multicall2 wrapper.

    - Uses tryAggregate(requireSuccess=false) to tolerate per-call failures.
    - Falls back to sequential if contract call fails.
    """

    MULTICALL2_ADDRESS = Web3.to_checksum_address("0x5BA1e12693DC8F9c48aAD8770482f4739bEeD696")

    ABI = [
        {
            "inputs": [
                {"internalType": "bool", "name": "requireSuccess", "type": "bool"},
                {
                    "components": [
                        {"internalType": "address", "name": "target", "type": "address"},
                        {"internalType": "bytes", "name": "callData", "type": "bytes"},
                    ],
                    "internalType": "struct Multicall2.Call[]",
                    "name": "calls",
                    "type": "tuple[]",
                },
            ],
            "name": "tryAggregate",
            "outputs": [
                {
                    "components": [
                        {"internalType": "bool", "name": "success", "type": "bool"},
                        {"internalType": "bytes", "name": "returnData", "type": "bytes"},
                    ],
                    "internalType": "struct Multicall2.Result[]",
                    "name": "returnData",
                    "type": "tuple[]",
                }
            ],
            "stateMutability": "payable",
            "type": "function",
        }
    ]

    def __init__(self, w3: Web3):
        self.w3 = w3
        try:
            self.contract = self.w3.eth.contract(address=self.MULTICALL2_ADDRESS, abi=self.ABI)
        except Exception:
            self.contract = None

    def try_aggregate(self, calls: List[Tuple[str, bytes]]) -> Optional[List[Tuple[bool, bytes]]]:
        try:
            if not self.contract:
                return None
            import time as _t
            retries = 1
            backoff = 0.2
            last_exc = None
            for attempt in range(retries + 1):
                try:
                    res = self.contract.functions.tryAggregate(False, [(t, cd) for (t, cd) in calls]).call()
                    out: List[Tuple[bool, bytes]] = []
                    for item in res:
                        try:
                            out.append((bool(item[0]), bytes(item[1])))
                        except Exception:
                            out.append((False, b""))
                    return out
                except Exception as e:
                    last_exc = e
                    if attempt < retries:
                        _t.sleep(backoff)
                        backoff = min(1.0, backoff * 2)
                        continue
                    raise
        except Exception as e:
            logger.debug(f"Multicall tryAggregate failed: {e}")
            return None

    def batch_get_pairs(self, factory_contract, pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Optional[str]]:
        """Batch getPair for Uniswap V2 factory.

        Returns mapping from sorted(lower) token pair key to pair address or None.
        """
        out: Dict[Tuple[str, str], Optional[str]] = {}
        try:
            calls: List[Tuple[str, bytes]] = []
            for a, b in pairs:
                try:
                    data = factory_contract.encodeABI(fn_name='getPair', args=[a, b])
                    calls.append((factory_contract.address, bytes.fromhex(data[2:])))
                except Exception:
                    # skip malformed
                    calls.append((factory_contract.address, b""))
            res = self.try_aggregate(calls)
            if not res or len(res) != len(pairs):
                return out
            for (a, b), (ok, r) in zip(pairs, res):
                key = tuple(sorted((a.lower(), b.lower())))
                if not ok or not r:
                    out[key] = None
                    continue
                try:
                    # decode via factory contract function ABI
                    decoded = factory_contract.decode_function_output('getPair', r)
                    addr = decoded[0] if isinstance(decoded, (list, tuple)) else decoded
                    if isinstance(addr, str) and int(addr, 16) != 0:
                        out[key] = addr
                    else:
                        out[key] = None
                except Exception:
                    out[key] = None
            return out
        except Exception as e:
            logger.debug(f"batch_get_pairs fallback due to error: {e}")
            return out

    def batch_v2_core(self, pair_addresses: List[str]) -> Dict[str, Dict[str, Optional[object]]]:
        """Batch fetch token0, token1, getReserves for V2 pair addresses.

        Returns mapping: pool_addr -> { 't0': str|None, 't1': str|None, 'r0': int|None, 'r1': int|None }
        """
        out: Dict[str, Dict[str, Optional[object]]] = {}
        if not pair_addresses:
            return out
        try:
            # Function selectors
            sel_token0 = bytes.fromhex('0dfe1681')
            sel_token1 = bytes.fromhex('d21220a7')
            sel_getReserves = bytes.fromhex('0902f1ac')
            calls: List[Tuple[str, bytes]] = []
            for addr in pair_addresses:
                calls.append((addr, sel_token0))
                calls.append((addr, sel_token1))
                calls.append((addr, sel_getReserves))
            res = self.try_aggregate(calls)
            if not res or len(res) != len(calls):
                return out
            for idx, addr in enumerate(pair_addresses):
                t0_ok, t0_data = res[idx*3 + 0]
                t1_ok, t1_data = res[idx*3 + 1]
                gr_ok, gr_data = res[idx*3 + 2]
                def _decode_addr(data: bytes) -> Optional[str]:
                    try:
                        if not data or len(data) < 32:
                            return None
                        # last 20 bytes
                        raw = data[-20:]
                        return Web3.to_checksum_address('0x' + raw.hex())
                    except Exception:
                        return None
                def _decode_reserves(data: bytes) -> Tuple[Optional[int], Optional[int]]:
                    try:
                        if not data or len(data) < 32*3:
                            return None, None
                        r0 = int.from_bytes(data[0:32], 'big')
                        r1 = int.from_bytes(data[32:64], 'big')
                        return r0, r1
                    except Exception:
                        return None, None
                t0 = _decode_addr(t0_data) if t0_ok else None
                t1 = _decode_addr(t1_data) if t1_ok else None
                r0, r1 = _decode_reserves(gr_data) if gr_ok else (None, None)
                out[Web3.to_checksum_address(addr)] = {'t0': t0, 't1': t1, 'r0': r0, 'r1': r1}
            return out
        except Exception as e:
            logger.debug(f"batch_v2_core failed: {e}")
            return out

    def batch_erc20_decimals(self, token_addresses: List[str]) -> Dict[str, int]:
        """Batch fetch ERC20 decimals for given token addresses. Defaults to 18 on failure."""
        out: Dict[str, int] = {}
        if not token_addresses:
            return out
        try:
            sel_decimals = bytes.fromhex('313ce567')
            calls: List[Tuple[str, bytes]] = [(addr, sel_decimals) for addr in token_addresses]
            res = self.try_aggregate(calls)
            if not res or len(res) != len(calls):
                for addr in token_addresses:
                    out[Web3.to_checksum_address(addr)] = 18
                return out
            for addr, (ok, data) in zip(token_addresses, res):
                dec = 18
                if ok and data and len(data) >= 32:
                    try:
                        dec = int.from_bytes(data[-32:], 'big')
                        if dec <= 0 or dec > 36:
                            dec = 18
                    except Exception:
                        dec = 18
                out[Web3.to_checksum_address(addr)] = dec
            return out
        except Exception as e:
            logger.debug(f"batch_erc20_decimals failed: {e}")
            for addr in token_addresses:
                out[Web3.to_checksum_address(addr)] = 18
            return out

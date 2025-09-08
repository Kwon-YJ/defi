from typing import Dict
from web3 import Web3

_DECIMALS_CACHE: Dict[str, int] = {}


def get_decimals(w3: Web3, token: str, default: int = 18) -> int:
    """Get ERC-20 decimals with simple in-memory cache.

    Falls back to default (18) if RPC call fails or w3 is not connected.
    """
    try:
        key = token.lower()
        if key in _DECIMALS_CACHE:
            return _DECIMALS_CACHE[key]
        if not w3 or not w3.is_connected():
            return default
        abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function",
            }
        ]
        c = w3.eth.contract(address=token, abi=abi)
        v = int(c.functions.decimals().call())
        _DECIMALS_CACHE[key] = v
        return v
    except Exception:
        return default


def normalize_amount(raw: int | float, decimals: int) -> float:
    """Convert a raw on-chain amount to human units using decimals."""
    try:
        return float(raw) / float(10 ** decimals)
    except Exception:
        return 0.0


def normalize_reserves(r0: int | float, d0: int, r1: int | float, d1: int) -> tuple[float, float]:
    """Normalize two reserves to human units using token decimals."""
    return normalize_amount(r0, d0), normalize_amount(r1, d1)


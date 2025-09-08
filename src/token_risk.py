import json
import os
from functools import lru_cache
from typing import Set


DEFAULT_FOT: Set[str] = set()
DEFAULT_REBASE: Set[str] = set()


@lru_cache(maxsize=1)
def _load_risk_sets() -> tuple[Set[str], Set[str]]:
    fot: Set[str] = set(DEFAULT_FOT)
    rebase: Set[str] = set(DEFAULT_REBASE)
    path = os.path.join('config', 'token_risks.json')
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            for addr in data.get('fee_on_transfer', []) or []:
                fot.add(str(addr).lower())
            for addr in data.get('rebase', []) or []:
                rebase.add(str(addr).lower())
    except Exception:
        pass
    return fot, rebase


def is_fee_on_transfer(token: str) -> bool:
    if not token:
        return False
    fot, _ = _load_risk_sets()
    return token.lower() in fot


def is_rebase(token: str) -> bool:
    if not token:
        return False
    _, rebase = _load_risk_sets()
    return token.lower() in rebase


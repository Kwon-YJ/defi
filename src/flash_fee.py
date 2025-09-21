from typing import Optional, Tuple, Dict, List
import asyncio
import os
from web3 import Web3
from src.logger import setup_logger

logger = setup_logger(__name__)


def _aave_v3_pool_address() -> str:
    # Mainnet Aave v3 Pool; overrideable via env AAVE_V3_POOL
    return os.getenv('AAVE_V3_POOL', '0x87870Bca3F3c6bCde8E99C4F3f07fBBeD3e7E5bA')


def estimate_aave_v3_premium(w3: Web3, pool_addr: Optional[str] = None) -> float:
    """Return Aave v3 flashloan premium as fraction (e.g., 0.0005 for 0.05%).

    Fallback to 0.0009 if read fails.
    """
    pool = pool_addr or _aave_v3_pool_address()
    try:
        if not w3 or not getattr(w3, 'is_connected', lambda: False)():
            raise RuntimeError('no w3')
        abi = [{"name": "FLASHLOAN_PREMIUM_TOTAL", "inputs": [], "outputs": [{"type": "uint128", "name": ""}], "stateMutability": "view", "type": "function"}]
        c = w3.eth.contract(address=pool, abi=abi)
        val = c.functions.FLASHLOAN_PREMIUM_TOTAL().call()
        # Aave v3 returns in bps (1e2?) or 1e4? Total premium is usually in bps (e.g., 9 = 0.09%).
        try:
            bps = int(val)
            if 0 < bps < 10000:
                return float(bps) / 10000.0
            # If looks like ray (1e27), convert heuristically
            if bps > 10000:
                return float(bps) / 1e6 / 10000.0
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"Aave v3 premium read failed: {e}")
    # fallback typical 0.09%
    return 0.0009


def estimate_fee_eth(provider: str, amount_eth: float, w3: Optional[Web3] = None) -> float:
    p = (provider or 'aave').lower()
    if p == 'aave':
        prem = 0.0009
        try:
            prem = estimate_aave_v3_premium(w3)
        except Exception:
            pass
        return max(0.0, amount_eth * prem)
    elif p == 'dydx':
        # dYdX older SoloMargin flash loans can be fee-free; keep configurable fallback
        try:
            fee_pct = float(os.getenv('DYDX_FLASH_FEE_PCT', '0.0'))
        except Exception:
            fee_pct = 0.0
        return max(0.0, amount_eth * fee_pct)
    else:
        try:
            fee_pct = float(os.getenv('FLASH_FEE_PCT_DEFAULT', '0.001'))  # 0.1%
        except Exception:
            fee_pct = 0.001
        return max(0.0, amount_eth * fee_pct)


def choose_best_provider_and_amount(opportunity, w3: Optional[Web3] = None) -> Dict:
    """Dry-run helper to choose provider and flash amount.

    - Evaluates provider in {aave, dydx} with a coarse grid of amounts around required_capital.
    - Uses SimulationExecutor to get path profit, then subtracts flash fee (+ optional extra gas).
    - Returns summary dict with best choice and breakdown.
    """
    from src.trade_executor import SimulationExecutor
    sim = SimulationExecutor(w3 or Web3())
    base = float(getattr(opportunity, 'required_capital', 0.0) or 0.0)
    if base <= 0:
        base = 1.0
    grid = [0.5, 0.75, 1.0, 1.25, 1.5]
    try:
        extra_gas_eth = float(os.getenv('FLASH_EXTRA_GAS_ETH', '0.0002'))  # ~200k at ~10-15 gwei
    except Exception:
        extra_gas_eth = 0.0002
    providers = ['aave', 'dydx']
    best = None  # (net_eth, provider, amount_eth, details)
    details: List[Dict] = []
    for prov in providers:
        for m in grid:
            amount_eth = max(0.01, base * m)
            # clone-like modify: we assume simulate_arbitrage reads required_capital only
            class _Tmp:
                pass
            tmp = _Tmp()
            for k in ('path','edges','profit_ratio','estimated_profit','gas_cost','net_profit','confidence','expected_value','sandwich_risk','mev_bribe_est','priority_fee_suggested'):
                if hasattr(opportunity, k):
                    setattr(tmp, k, getattr(opportunity, k))
            tmp.required_capital = amount_eth
            res = {}
            try:
                res = asyncio.get_event_loop().run_until_complete(sim.simulate_arbitrage(tmp))  # in sync context fallback
            except RuntimeError:
                # if already in loop, use a minimal sync approximation
                res = {'net_profit': float(getattr(opportunity, 'net_profit', 0.0) or 0.0) * (amount_eth / base), 'profit_ratio': getattr(opportunity, 'profit_ratio', 1.0)}
            path_net = float(res.get('net_profit', 0.0) or 0.0)
            fee = estimate_fee_eth(prov, amount_eth, w3)
            net = path_net - fee - extra_gas_eth
            details.append({'provider': prov, 'amount_eth': amount_eth, 'path_net_eth': path_net, 'fee_eth': fee, 'extra_gas_eth': extra_gas_eth, 'net_eth': net})
            if (best is None) or (net > best[0]):
                best = (net, prov, amount_eth)
    best_detail = [d for d in details if d['provider'] == best[1] and abs(d['amount_eth'] - best[2]) < 1e-9]
    return {
        'best_provider': best[1] if best else None,
        'best_amount_eth': best[2] if best else None,
        'best_net_eth': best[0] if best else None,
        'samples': details,
        'base_required_eth': base,
    }

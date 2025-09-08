"""Utilities for standardized synthetic token IDs used in the graph.

We use string identifiers for synthetic tokens that don't have a canonical ERC-20
address in our simplified modeling (e.g., Uniswap V3 LP, Balancer BPT, debt tokens).

Format conventions:
- LP tokens
  - Uniswap V2 LP:        lp:v2:<pair_address>
  - Uniswap V3 LP (synth):lp:v3:<pool_address>
  - Curve LP (synth):     lp:curve:<pool_address>
  - Balancer BPT (synth): bpt:<pool_address>

- Debt tokens
  - Aave variable debt:   debt:aave:<underlying_address>
  - Compound debt:        debt:compound:<underlying_address>
"""

def lp_v2(pair_address: str) -> str:
    return f"lp:v2:{pair_address}"


def lp_v3(pool_address: str) -> str:
    return f"lp:v3:{pool_address}"


def lp_curve(pool_address: str) -> str:
    return f"lp:curve:{pool_address}"


def bpt(pool_address: str) -> str:
    return f"bpt:{pool_address}"


def debt_aave(underlying: str) -> str:
    return f"debt:aave:{underlying}"


def debt_compound(underlying: str) -> str:
    return f"debt:compound:{underlying}"


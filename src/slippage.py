from typing import Optional


def amount_out_cpmm(amount_in: float, reserve_in: float, reserve_out: float, fee_fraction: float) -> float:
    if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
        return 0.0
    ai = amount_in * max(0.0, 1.0 - fee_fraction)
    return (ai * reserve_out) / (reserve_in + ai)


def amount_out_uniswap_v2(amount_in: float, reserve_in: float, reserve_out: float, fee_fraction: float) -> float:
    return amount_out_cpmm(amount_in, reserve_in, reserve_out, fee_fraction)


def amount_out_balancer_weighted(amount_in: float, reserve_in: float, reserve_out: float,
                                 w_in: float, w_out: float, fee_fraction: float) -> float:
    if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
        return 0.0
    ai = amount_in * max(0.0, 1.0 - fee_fraction)
    if w_in <= 0 or w_out <= 0:
        return amount_out_cpmm(amount_in, reserve_in, reserve_out, fee_fraction)
    ratio = reserve_in / (reserve_in + ai)
    try:
        power = w_in / w_out
    except Exception:
        power = 1.0
    return reserve_out * (1.0 - (ratio ** power))


def amount_out_curve_stable_approx(amount_in: float, reserve_in: float, reserve_out: float,
                                   fee_fraction: float, amp: float = 100.0) -> float:
    """Approximate stableswap with reduced price impact by amplification factor.

    We approximate by dividing the input by amp before applying CPMM; amp >> 1 lowers impact.
    """
    if amp <= 0:
        amp = 100.0
    return amount_out_cpmm(amount_in / amp, reserve_in, reserve_out, fee_fraction) * amp


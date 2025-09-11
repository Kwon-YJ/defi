from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

from web3 import Web3
from src.logger import setup_logger
from config.config import config

logger = setup_logger(__name__)


@dataclass
class DyDxFlashPlan:
    asset: str
    amount_wei: int
    solo_margin: str
    steps: List[Dict[str, Any]]


class DyDxFlashExecutor:
    """dYdX SoloMargin 기반 플래시 론 실행/시뮬 스캐폴딩.

    참고: 실제 on-chain 실행에는 SoloMargin의 operate(Account.Info[], Actions.ActionArgs[])를 구성해야 하며,
         Withdraw -> Call -> Deposit 시퀀스를 통해 자금 대여/사용/상환을 수행합니다. 여기서는 시뮬/계획 산출까지만 제공합니다.
    """

    def __init__(self, w3: Web3 | None = None):
        self.w3 = w3 or Web3()
        self.solo = getattr(config, 'dydx_solo_margin', '')

    def build_plan(self, opportunity) -> DyDxFlashPlan:
        # 첫 토큰을 대여 자산으로 가정
        base_token = opportunity.path[0]
        # 간단화: 기회에 기록된 required_capital를 대여
        amount_wei = int(float(opportunity.required_capital) * 1e18)
        steps = []
        for i, edge in enumerate(opportunity.edges):
            steps.append({
                'idx': i + 1,
                'dex': edge.dex,
                'from': edge.token_in if hasattr(edge, 'token_in') else opportunity.path[i],
                'to': edge.token_out if hasattr(edge, 'token_out') else opportunity.path[i+1],
                'fee': edge.fee,
            })
        return DyDxFlashPlan(asset=str(base_token), amount_wei=amount_wei, solo_margin=self.solo, steps=steps)

    async def simulate(self, opportunity) -> Dict[str, Any]:
        """dYdX 플래시 대여를 사용한다는 가정 하에 기회 시뮬.

        실제 스왑 시뮬은 기존 SimulationExecutor에 위임하는 편이 적합하지만,
        여기서는 dYdX 경로/상환 논리가 적용됨을 명시적으로 로깅합니다.
        """
        from src.trade_executor import SimulationExecutor
        sim = SimulationExecutor(self.w3)
        result = await sim.simulate_arbitrage(opportunity)
        plan = self.build_plan(opportunity)
        result.update({
            'provider': 'dydx',
            'solo_margin': plan.solo_margin,
            'flash_asset': plan.asset,
            'flash_amount_wei': plan.amount_wei,
        })
        logger.info(f"[dYdX] simulate plan: asset={plan.asset}, amountWei={plan.amount_wei}, solo={plan.solo_margin}")
        return result

    async def execute_onchain(self, opportunity) -> str:
        """온체인 실행 스캐폴드. 실제 operate() 호출 구성은 추후 확장.

        현재는 미구현 상태로 예외를 발생시켜 안전하게 빠져나옵니다.
        """
        raise NotImplementedError("dYdX on-chain execute는 추후 SoloMargin operate 구성 후 지원")


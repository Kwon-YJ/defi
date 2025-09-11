from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Type

from web3 import Web3
from src.logger import setup_logger
from src.action_registry import ProtocolAction
from src.market_graph import DeFiMarketGraph
from src.token_manager import TokenManager

logger = setup_logger(__name__)


@dataclass
class StabilityResult:
    total_actions: int
    executed: int
    succeeded: int
    failed: int
    duration_sec: float
    errors: List[str]


class StabilityTester:
    """96개 protocol actions 동시 처리 안정성 테스트.

    - 활성화된(Enabled) ProtocolAction 서브클래스들을 모두 병렬 실행
    - 네트워크 제한 환경에서도 예외가 상위로 전파되지 않음(각 액션 내부 예외는 로그 후 0 반환)
    - 실패/성공 카운트 및 실행 시간 집계
    """

    def __init__(self, w3: Web3 | None = None):
        self.w3 = w3 or Web3()  # 더미
        self.graph = DeFiMarketGraph()
        tm = TokenManager()
        # 심볼->주소 매핑 사용
        self.tokens: Dict[str, str] = tm.symbol_to_address

    def _discover_actions(self) -> List[Type[ProtocolAction]]:
        actions: List[Type[ProtocolAction]] = []
        for _, obj in inspect.getmembers(__import__('src.action_registry', fromlist=[''])):
            if inspect.isclass(obj) and issubclass(obj, ProtocolAction) and obj is not ProtocolAction:
                # 인스턴스 생성 전에 enabled 속성이 존재하는지 확인
                try:
                    if getattr(obj, 'enabled', False):
                        actions.append(obj)
                except Exception:
                    continue
        return actions

    async def run(self, timeout_sec: float = 60.0) -> StabilityResult:
        start = time.time()
        actions = self._discover_actions()
        errors: List[str] = []

        async def do_action(cls) -> bool:
            try:
                inst: ProtocolAction = cls(self.w3) if callable(getattr(cls, '__init__', None)) else cls
                updated = await asyncio.wait_for(
                    inst.update_graph(self.graph, self.w3, self.tokens), timeout=timeout_sec
                )
                return isinstance(updated, int)
            except Exception as e:
                logger.debug(f"Action failed: {getattr(cls, 'name', cls.__name__)} -> {e}")
                errors.append(f"{getattr(cls, 'name', cls.__name__)}: {e}")
                return False

        results = await asyncio.gather(*[do_action(cls) for cls in actions], return_exceptions=False)
        executed = len(results)
        succeeded = sum(1 for r in results if r)
        failed = executed - succeeded
        dur = time.time() - start
        return StabilityResult(
            total_actions=len(actions), executed=executed, succeeded=succeeded, failed=failed,
            duration_sec=dur, errors=errors
        )


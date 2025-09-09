from typing import Dict, List, Optional
import json
import os
from src.logger import setup_logger

logger = setup_logger(__name__)


def paper_25_symbols() -> List[str]:
    """논문에서 사용한 25개 자산(심볼) 목록.

    주: 원문에서는 ETH를 기준으로 24개 토큰과의 페어가 제시되어 총 25개 자산으로 구성됨.
    목록은 심볼 기준이며 주소는 JSON 구성에서 로드한다.
    """
    return [
        # ETH는 WETH 주소로 처리
        "ETH",
        # 나머지 24개 (알파벳순 아님, 논문 표기 순서 참고)
        "AMN", "AMPL", "ANT", "BAT", "BNT", "DAI", "DATA", "ENJ",
        "FXC", "GNO", "HEDG", "KNC", "MANA", "MKR", "POA20", "RCN",
        "RDN", "RLC", "SAI", "SAN", "SNT", "TKN", "TRST", "UBT",
    ]


def load_paper_25_addresses(json_path: str = os.path.join('config', 'paper_assets_25.json')) -> Dict[str, str]:
    """config/paper_assets_25.json에서 심볼→주소 매핑을 로드한다.

    - 미기재/잘못된 항목은 제외한다.
    - ETH는 WETH 주소를 사용하도록 권장한다.
    """
    mapping: Dict[str, str] = {}
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        for sym in paper_25_symbols():
            ent = data.get(sym)
            if not isinstance(ent, dict):
                continue
            addr = ent.get('address')
            if isinstance(addr, str) and addr.startswith('0x') and len(addr) == 42:
                mapping[sym] = addr
            else:
                logger.warning(f"paper_25: 주소 누락/형식 오류(sym={sym}). JSON에서 보완 필요")
    except FileNotFoundError:
        logger.warning(f"paper_25 자산 구성 파일을 찾을 수 없습니다: {json_path}")
    except Exception as e:
        logger.error(f"paper_25 자산 로드 실패: {e}")
    return mapping


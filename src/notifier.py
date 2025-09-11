import subprocess
import shlex
import os
from typing import Optional

from src.logger import setup_logger

logger = setup_logger(__name__)


class Notifier:
    """간단한 알림 발행기.

    우선순위: telegram_send.py 스크립트가 있으면 호출 → 없으면 무시.
    파일/콘솔 알림은 monitor가 처리하므로 여기서는 외부 채널만 시도.
    """

    def __init__(self, script_path: str = 'telegram_send.py'):
        self.script_path = script_path

    def send(self, message: str) -> bool:
        if not os.path.exists(self.script_path):
            logger.debug("telegram_send.py 미존재 – 외부 알림 건너뜀")
            return False
        try:
            cmd = f"python3 {shlex.quote(self.script_path)} --msg {shlex.quote(message)}"
            subprocess.run(cmd, shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            logger.debug(f"텔레그램 알림 실패: {e}")
            return False


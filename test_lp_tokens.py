#!/usr/bin/env python3

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.token_manager import TokenManager
from src.logger import setup_logger

logger = setup_logger(__name__)

def test_lp_tokens():
    """LP tokens 구현 테스트"""
    logger.info("=== LP Tokens Implementation Test ===")
    
    # TokenManager 초기화
    token_manager = TokenManager()
    
    # LP tokens 검증
    lp_tokens = [
        # Uniswap V2 LP tokens
        "WETH-USDC", "WETH-USDT", "WETH-DAI", "WBTC-WETH", 
        "UNI-WETH", "COMP-WETH", "AAVE-WETH", "SUSHI-WETH", 
        "MKR-WETH", "USDC-USDT",
        
        # Curve LP tokens
        "3CRV", "crvUSDC", "yDAI+yUSDC+yUSDT+yTUSD",
        "GUSD3CRV", "HUSD3CRV", "USDK3CRV", "USDN3CRV", 
        "USDP3CRV", "crvRenWSBTC", "crvHBTC"
    ]
    
    logger.info(f"Testing {len(lp_tokens)} LP tokens...")
    
    success_count = 0
    for symbol in lp_tokens:
        address = token_manager.get_address_by_symbol(symbol)
        if address:
            token_info = token_manager.tokens[address]
            logger.info(f"✅ {symbol}: {token_info.name} ({address[:10]}...)")
            success_count += 1
        else:
            logger.error(f"❌ {symbol}: Address not found")
    
    logger.info(f"\n=== LP Token Registration Results ===")
    logger.info(f"Successfully registered: {success_count}/{len(lp_tokens)} LP tokens")
    logger.info(f"Total tokens in system: {len(token_manager.tokens)}")
    
    # LP token trading pairs 테스트
    trading_pairs = token_manager.get_major_trading_pairs()
    lp_pairs = []
    
    for addr0, addr1 in trading_pairs:
        token0 = token_manager.tokens.get(addr0)
        token1 = token_manager.tokens.get(addr1) 
        if token0 and token1:
            # LP 토큰이 포함된 쌍 찾기
            if ("-" in token0.symbol or "CRV" in token0.symbol or "crv" in token0.symbol or 
                "yDAI+" in token0.symbol or "-" in token1.symbol or "CRV" in token1.symbol or 
                "crv" in token1.symbol or "yDAI+" in token1.symbol):
                lp_pairs.append((token0.symbol, token1.symbol))
    
    logger.info(f"\n=== LP Token Trading Pairs ===")
    logger.info(f"Found {len(lp_pairs)} LP token trading pairs")
    
    # 주요 LP 토큰 쌍들 출력
    sample_pairs = lp_pairs[:20]  # 처음 20개만 출력
    for token0, token1 in sample_pairs:
        logger.info(f"  {token0} ↔ {token1}")
    
    if len(lp_pairs) > 20:
        logger.info(f"  ... and {len(lp_pairs) - 20} more pairs")
    
    # 검증 결과
    if success_count == len(lp_tokens) and len(lp_pairs) > 0:
        logger.info(f"\n✅ LP Tokens Implementation SUCCESS!")
        logger.info(f"   - All {len(lp_tokens)} LP tokens registered")
        logger.info(f"   - {len(lp_pairs)} LP trading pairs available")
        logger.info(f"   - Total assets: {len(token_manager.tokens)}")
        return True
    else:
        logger.error(f"\n❌ LP Tokens Implementation FAILED!")
        return False

if __name__ == "__main__":
    success = test_lp_tokens()
    sys.exit(0 if success else 1)
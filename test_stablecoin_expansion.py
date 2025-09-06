#!/usr/bin/env python3
"""
Test script to validate USDC, USDT, DAI stablecoin expansion
This validates the completion of TODO item: "Stablecoins: USDC, USDT, DAI (현재 있음, 확장 필요)"
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from token_manager import TokenManager
from logger import setup_logger

logger = setup_logger(__name__)

async def test_stablecoin_expansion():
    """Test stablecoin expansion - USDC, USDT, DAI"""
    
    logger.info("🧪 Testing stablecoin expansion (USDC, USDT, DAI)...")
    
    # Initialize token manager
    token_manager = TokenManager()
    
    # Test stablecoins
    stablecoins_to_test = {
        'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',    # Already existed
        'USDC': '0xA0b86a33E6441946e15b3C1F5d44F7c0e3A1b82C',   # Newly added  
        'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7'    # Newly added
    }
    
    logger.info("✅ Testing stablecoin registration...")
    all_passed = True
    
    for symbol, expected_address in stablecoins_to_test.items():
        # Test symbol -> address mapping
        actual_address = token_manager.get_address_by_symbol(symbol)
        if actual_address and actual_address.lower() == expected_address.lower():
            logger.info(f"✅ {symbol}: {actual_address}")
        else:
            logger.error(f"❌ {symbol}: Expected {expected_address.lower()}, got {actual_address}")
            all_passed = False
        
        # Test token info retrieval (check in registered tokens)
        token_info = None
        for addr, info in token_manager.tokens.items():
            if addr.lower() == expected_address.lower():
                token_info = info
                break
        
        if token_info and token_info.symbol == symbol:
            logger.info(f"✅ {symbol} token info: {token_info.name} ({token_info.decimals} decimals)")
        else:
            logger.error(f"❌ {symbol} token info retrieval failed")
            all_passed = False
    
    # Test stablecoin arbitrage pairs
    logger.info("✅ Testing stablecoin arbitrage pairs...")
    trading_pairs = token_manager.get_major_trading_pairs()
    
    expected_stablecoin_pairs = [
        ('USDC', 'USDT'), ('USDT', 'USDC'),
        ('DAI', 'USDC'), ('USDC', 'DAI'), 
        ('DAI', 'USDT'), ('USDT', 'DAI')
    ]
    
    # Convert to addresses for checking
    expected_address_pairs = []
    for s1, s2 in expected_stablecoin_pairs:
        addr1 = token_manager.get_address_by_symbol(s1)
        addr2 = token_manager.get_address_by_symbol(s2)
        if addr1 and addr2:
            expected_address_pairs.append((addr1, addr2))
    
    found_pairs = 0
    for expected_pair in expected_address_pairs:
        if expected_pair in trading_pairs:
            found_pairs += 1
            logger.info(f"✅ Found stablecoin pair: {expected_pair}")
    
    logger.info(f"📊 Stablecoin pairs found: {found_pairs}/{len(expected_address_pairs)}")
    
    if found_pairs == len(expected_address_pairs):
        logger.info("✅ All stablecoin arbitrage pairs are properly configured!")
    else:
        logger.error("❌ Missing some stablecoin arbitrage pairs")
        all_passed = False
    
    # Summary
    total_tokens = len(token_manager.tokens)
    logger.info(f"📊 Total tokens registered: {total_tokens}")
    
    if all_passed:
        logger.info("🎉 STABLECOIN EXPANSION COMPLETED SUCCESSFULLY!")
        logger.info("✅ TODO Item Complete: 'Stablecoins: USDC, USDT, DAI (현재 있음, 확장 필요)'")
        return True
    else:
        logger.error("❌ Stablecoin expansion has issues")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_stablecoin_expansion())
    sys.exit(0 if success else 1)
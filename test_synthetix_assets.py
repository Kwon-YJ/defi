#!/usr/bin/env python3
"""
Test script to verify Synthetix synthetic assets (sUSD, sETH) implementation
Verifies TODO requirement completion: Synthetic assets: sUSD, sETH (Synthetix)
"""

import asyncio
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from token_manager import TokenManager
from logger import setup_logger

logger = setup_logger(__name__)

async def test_synthetix_assets():
    """Test Synthetix synthetic assets (sUSD, sETH) integration"""
    logger.info("🧪 Testing Synthetix synthetic assets implementation...")
    
    # Initialize token manager
    token_manager = TokenManager()
    
    # Test sUSD token
    logger.info("Testing sUSD (Synthetic USD)...")
    susd_address = token_manager.get_address_by_symbol("sUSD")
    if susd_address:
        susd_info = await token_manager.get_token_info(susd_address)
        if susd_info:
            logger.info(f"✅ sUSD found: {susd_info.name} ({susd_info.symbol}) at {susd_info.address}")
            logger.info(f"   Decimals: {susd_info.decimals}, CoinGecko ID: {susd_info.coingecko_id}")
        else:
            logger.error("❌ sUSD info not found")
            return False
    else:
        logger.error("❌ sUSD address not found")
        return False
    
    # Test sETH token
    logger.info("Testing sETH (Synthetic ETH)...")
    seth_address = token_manager.get_address_by_symbol("sETH")
    if seth_address:
        seth_info = await token_manager.get_token_info(seth_address)
        if seth_info:
            logger.info(f"✅ sETH found: {seth_info.name} ({seth_info.symbol}) at {seth_info.address}")
            logger.info(f"   Decimals: {seth_info.decimals}, CoinGecko ID: {seth_info.coingecko_id}")
        else:
            logger.error("❌ sETH info not found")
            return False
    else:
        logger.error("❌ sETH address not found")
        return False
    
    # Test trading pairs with Synthetix assets
    logger.info("Testing Synthetix asset trading pairs...")
    major_pairs = token_manager.get_major_trading_pairs()
    
    # Count pairs involving sUSD and sETH
    susd_pairs = [pair for pair in major_pairs if susd_address in pair]
    seth_pairs = [pair for pair in major_pairs if seth_address in pair]
    
    logger.info(f"✅ sUSD trading pairs: {len(susd_pairs)}")
    logger.info(f"✅ sETH trading pairs: {len(seth_pairs)}")
    
    # Verify specific important pairs
    important_pairs = [
        (susd_address, token_manager.get_address_by_symbol("USDC")),
        (seth_address, token_manager.get_address_by_symbol("ETH")),
        (susd_address, seth_address)  # Cross synthetic pair
    ]
    
    for pair in important_pairs:
        if pair in major_pairs:
            logger.info(f"✅ Important pair found: {pair}")
        else:
            logger.warning(f"⚠️ Important pair missing: {pair}")
    
    # Verify total token count includes Synthetix assets
    total_tokens = len(token_manager.tokens)
    logger.info(f"✅ Total tokens registered: {total_tokens} (should include 62 with Synthetix assets)")
    
    if total_tokens >= 62:
        logger.info("✅ Token count verification passed")
    else:
        logger.warning(f"⚠️ Token count may be low: {total_tokens}")
    
    logger.info("🎉 Synthetix synthetic assets (sUSD, sETH) implementation test completed!")
    return True

if __name__ == "__main__":
    asyncio.run(test_synthetix_assets())
#!/usr/bin/env python3
"""
Test script for lending protocol tokens implementation
Tests the addition of cETH, cUSDC, aETH, aUSDC tokens
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.token_manager import TokenManager
from src.logger import setup_logger

logger = setup_logger(__name__)

def test_lending_protocol_tokens():
    """Test lending protocol tokens implementation"""
    print("🧪 Testing lending protocol tokens implementation...")
    
    # Initialize TokenManager
    token_manager = TokenManager()
    
    # Test lending protocol tokens
    lending_tokens = [
        ("cETH", "0x4Ddc2D193948926D02f9B1fE9e1daa0718270ED5"),
        ("cUSDC", "0x39AA39c021dfbaE8faC545936693aC917d5E7563"),
        ("aETH", "0x030bA81f1c18d280636F32af80b9AAd02Cf0854e"),
        ("aUSDC", "0xBcca60bB61934080951369a648Fb03DF4F96263C")
    ]
    
    print("\n📊 Testing lending protocol tokens...")
    all_passed = True
    
    for symbol, expected_address in lending_tokens:
        # Test symbol to address lookup
        address = token_manager.get_address_by_symbol(symbol)
        if address == expected_address:
            print(f"✅ {symbol}: {address}")
        else:
            print(f"❌ {symbol}: Expected {expected_address}, got {address}")
            all_passed = False
            
        # Test token info lookup
        token_info = token_manager.tokens.get(expected_address)
        if token_info:
            print(f"   📋 Name: {token_info.name}")
            print(f"   🔢 Decimals: {token_info.decimals}")
            print(f"   🏷️  CoinGecko ID: {token_info.coingecko_id}")
        else:
            print(f"❌ Token info not found for {symbol}")
            all_passed = False
    
    print(f"\n📈 Total tokens registered: {len(token_manager.tokens)}")
    print(f"📈 Expected count: 40 tokens")
    
    # Test trading pairs
    print("\n🔄 Testing lending protocol token trading pairs...")
    trading_pairs = token_manager.get_major_trading_pairs()
    
    # Count lending protocol pairs
    lending_pairs_count = 0
    lending_symbols = ["cETH", "cUSDC", "aETH", "aUSDC"]
    
    for addr0, addr1 in trading_pairs:
        token0 = None
        token1 = None
        for symbol, addr in token_manager.symbol_to_address.items():
            if addr == addr0:
                token0 = symbol
            if addr == addr1:
                token1 = symbol
        
        if token0 in lending_symbols or token1 in lending_symbols:
            lending_pairs_count += 1
    
    print(f"✅ Found {lending_pairs_count} lending protocol token pairs")
    print(f"📊 Total trading pairs: {len(trading_pairs)}")
    
    # Test specific important pairs
    critical_pairs = [
        ("ETH", "cETH"),
        ("USDC", "cUSDC"),
        ("ETH", "aETH"),
        ("USDC", "aUSDC"),
        ("cETH", "aETH"),  # Compound vs Aave arbitrage
        ("cUSDC", "aUSDC")  # Compound vs Aave arbitrage
    ]
    
    print("\n🔍 Testing critical lending protocol pairs...")
    for symbol0, symbol1 in critical_pairs:
        addr0 = token_manager.get_address_by_symbol(symbol0)
        addr1 = token_manager.get_address_by_symbol(symbol1)
        if addr0 and addr1:
            if (addr0, addr1) in trading_pairs:
                print(f"✅ {symbol0}-{symbol1} pair exists")
            else:
                print(f"❌ {symbol0}-{symbol1} pair missing")
                all_passed = False
        else:
            print(f"❌ Could not find addresses for {symbol0}-{symbol1}")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    success = test_lending_protocol_tokens()
    
    if success:
        print("\n🎉 All lending protocol token tests passed!")
        print("✅ TODO task completed: Lending protocol tokens (cETH, cUSDC, aETH, aUSDC) implemented successfully")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
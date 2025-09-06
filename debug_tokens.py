#!/usr/bin/env python3
"""Debug script to check what tokens are loaded"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.token_manager import TokenManager

def debug_tokens():
    token_manager = TokenManager()
    
    print("=== All loaded tokens ===")
    for address, token in token_manager.tokens.items():
        print(f"{token.symbol}: {address}")
    
    print(f"\nTotal tokens: {len(token_manager.tokens)}")
    
    print("\n=== Symbol to Address mapping ===")
    for symbol, address in token_manager.symbol_to_address.items():
        print(f"{symbol}: {address}")
    
    print(f"\nTotal symbol mappings: {len(token_manager.symbol_to_address)}")
    
    # Test specific lending tokens
    lending_tokens = ["cETH", "cUSDC", "aETH", "aUSDC"]
    print("\n=== Lending token lookup test ===")
    for symbol in lending_tokens:
        address = token_manager.get_address_by_symbol(symbol)
        print(f"{symbol}: {address}")

if __name__ == "__main__":
    debug_tokens()
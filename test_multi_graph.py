#!/usr/bin/env python3
"""
Multi-graph 지원 기능 테스트
동일 토큰 쌍에서 여러 DEX edge 처리 검증
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.market_graph import DeFiMarketGraph
from src.logger import setup_logger

logger = setup_logger(__name__)

def test_multi_graph_support():
    """Multi-graph 지원 기능 테스트"""
    logger.info("=== Multi-graph 지원 기능 테스트 시작 ===")
    
    # MarketGraph 초기화
    graph = DeFiMarketGraph()
    
    # 동일한 토큰 쌍(WETH-USDC)에 여러 DEX 추가
    token_weth = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    token_usdc = "0xA0b86a33E6Ff4C0a14F9F4c8C5D2d9c9aD1b3F9B"
    
    # 1. Uniswap V2 풀 추가
    logger.info("1. Uniswap V2 풀 추가")
    graph.add_trading_pair(
        token_weth, token_usdc, "uniswap_v2",
        "0x397FF1542f962076d0BFE58eA045FfA2d347ACa0",  # Uniswap V2 WETH/USDC
        reserve0=1000.0,    # 1000 WETH
        reserve1=2000000.0, # 2M USDC (환율: 1 WETH = 2000 USDC)
        fee=0.003
    )
    
    # 2. Uniswap V3 풀 추가 (더 나은 환율)
    logger.info("2. Uniswap V3 풀 추가 (더 나은 환율)")
    graph.add_trading_pair(
        token_weth, token_usdc, "uniswap_v3",
        "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8",  # Uniswap V3 WETH/USDC
        reserve0=2000.0,    # 2000 WETH
        reserve1=4100000.0, # 4.1M USDC (환율: 1 WETH = 2050 USDC, 더 나은 환율)
        fee=0.0005
    )
    
    # 3. SushiSwap 풀 추가 (나쁜 환율)
    logger.info("3. SushiSwap 풀 추가 (나쁜 환율)")
    graph.add_trading_pair(
        token_weth, token_usdc, "sushiswap",
        "0x06da0fd433C1A5d7a4faa01111c044910A184553",  # SushiSwap WETH/USDC
        reserve0=500.0,     # 500 WETH
        reserve1=950000.0,  # 950K USDC (환율: 1 WETH = 1900 USDC, 나쁜 환율)
        fee=0.003
    )
    
    # Multi-graph 통계 확인
    logger.info("=== Multi-graph 통계 ===")
    multi_stats = graph.get_multi_graph_stats()
    for key, value in multi_stats.items():
        logger.info(f"{key}: {value}")
    
    # 토큰 쌍별 DEX 개수 확인
    dex_count = graph.get_dex_count_for_pair(token_weth, token_usdc)
    logger.info(f"WETH-USDC 쌍 지원 DEX 개수: {dex_count}")
    
    # 모든 edge 정보 확인 (환율 기준 정렬)
    logger.info("=== WETH -> USDC 모든 edge 정보 ===")
    all_edges = graph.get_all_edges(token_weth, token_usdc)
    for i, edge in enumerate(all_edges, 1):
        logger.info(f"{i}. DEX: {edge.get('dex', 'Unknown')}, "
                   f"환율: {edge.get('exchange_rate', 0):.2f}, "
                   f"수수료: {edge.get('fee', 0):.4f}, "
                   f"유동성: {edge.get('liquidity', 0):.0f}")
    
    # 최적 edge 정보 확인
    logger.info("=== 최적 edge 정보 ===")
    best_edge = graph.get_best_edge(token_weth, token_usdc)
    if best_edge:
        logger.info(f"최적 DEX: {best_edge.get('dex')}, "
                   f"환율: {best_edge.get('exchange_rate'):.2f}, "
                   f"수수료: {best_edge.get('fee'):.4f}")
    else:
        logger.warning("최적 edge를 찾을 수 없습니다")
    
    # 특정 DEX의 edge 제거 테스트
    logger.info("=== SushiSwap edge 제거 테스트 ===")
    removed = graph.remove_edge_by_dex(token_weth, token_usdc, "sushiswap")
    logger.info(f"SushiSwap edge 제거: {removed}")
    
    # 제거 후 통계 확인
    new_dex_count = graph.get_dex_count_for_pair(token_weth, token_usdc)
    logger.info(f"제거 후 WETH-USDC 쌍 지원 DEX 개수: {new_dex_count}")
    
    # 최종 그래프 통계
    logger.info("=== 최종 그래프 통계 ===")
    final_stats = graph.get_graph_stats()
    for key, value in final_stats.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {sub_value}")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("=== Multi-graph 지원 기능 테스트 완료 ===")
    return True

def main():
    """메인 함수"""
    try:
        success = test_multi_graph_support()
        if success:
            logger.info("✅ Multi-graph 테스트 성공")
        else:
            logger.error("❌ Multi-graph 테스트 실패")
            return 1
        return 0
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
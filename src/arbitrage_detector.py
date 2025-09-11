#!/usr/bin/env python3
"""
DeFi Arbitrage Detector - Main execution script
실시간 차익거래 기회 탐지 및 실행
"""

import asyncio
import os
import json
from typing import List, Dict
from datetime import datetime
from web3 import Web3
from src.market_graph import DeFiMarketGraph
from src.bellman_ford_arbitrage import BellmanFordArbitrage
from src.data_storage import DataStorage
from src.logger import setup_logger
from src.token_manager import TokenManager
from src.real_time_collector import RealTimeDataCollector

logger = setup_logger(__name__)

class ArbitrageDetector:
    def __init__(self):
        # --- Web3 Setup ---
        # Note: Ensure the ETH_WS_URL environment variable is set.
        self.w3 = Web3(Web3.AsyncWebsocketProvider(os.environ.get("ETH_WS_URL", "ws://localhost:8546")))
        
        # --- Contract ABI Loading ---
        self.uniswap_v2_factory_abi = self._load_abi('../abi/uniswap_v2_factory.json')
        self.uniswap_v2_pair_abi = self._load_abi('../abi/uniswap_v2_pair.json')
        self.erc20_abi = self._load_abi('../abi/erc20.json')

        # --- Core Components ---
        self.token_manager = TokenManager()
        self.market_graph = DeFiMarketGraph()
        self.bellman_ford = BellmanFordArbitrage(self.market_graph)
        self.storage = DataStorage()
        self.collector = RealTimeDataCollector()
        self.running = False
        self.updated_pools = set()
        
        # --- Token Configuration ---
        self.base_tokens = [self.token_manager.get_address_by_symbol("WETH")] # WETH as the primary base token
        self.token_decimals = {}

    def _load_abi(self, filepath: str) -> Dict:
        """Helper to load ABI from a file."""
        # Construct path relative to this file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, filepath)) as f:
            return json.load(f)

    async def _get_token_decimals(self, token_address: str) -> int:
        """Fetches and caches the decimals for a given token."""
        token_info = await self.token_manager.get_token_info(token_address)
        if token_info:
            return token_info.decimals
        
        logger.warning(f"토큰 decimals 조회 오류 {token_address}. 기본값 18 사용.")
        return 18

    async def _handle_sync_event(self, event):
        """Callback for Sync events."""
        pool_address = event['address']
        self.updated_pools.add(pool_address)

    async def start_detection(self):
        """
        Starts the arbitrage detection by listening for new blocks and handling them.
        """
        self.running = True
        logger.info("차익거래 탐지 시작 (이벤트 기반 업데이트)")

        # Subscribe to Sync events
        await self.collector.subscribe_to_logs(self._handle_sync_event)
        
        # Start the collector in the background
        collector_task = asyncio.create_task(self.collector.start_websocket_listener())

        # Main loop to process new blocks
        while self.running:
            try:
                # We can use the block notifier from the collector
                # or just process based on a timer if we are event-driven
                await asyncio.sleep(1) # Process every second
                if self.updated_pools:
                    await self._handle_new_block(None) # Pass a dummy block hash

            except Exception as e:
                logger.error(f"메인 루프 오류: {e}")
                await asyncio.sleep(10)


    async def _handle_new_block(self, block_hash: bytes):
        """
        Handles a new block by updating market data and searching for arbitrage.
        """
        try:
            # 1. Update market data based on the new block state
            await self._update_market_data()
            
            # 2. Perform local search for the best arbitrage opportunity
            await self._run_local_search()

        except Exception as e:
            logger.error(f"블록 처리 오류 ({block_hash.hex()}): {e}")

    async def _run_local_search(self):
        """
        Runs the local search loop: find all profitable opportunities, 
        execute them greedily, update graph, and repeat.
        """
        while True:
            # a. Search for opportunities in parallel across all base tokens
            logger.debug(f"{len(self.base_tokens)}개의 시작점에서 병렬 탐색 시작")
            tasks = [asyncio.to_thread(self.bellman_ford.find_negative_cycles, base_token) for base_token in self.base_tokens]
            results = await asyncio.gather(*tasks)

            all_opportunities = [opp for sublist in results for opp in sublist]
            
            # b. If no profitable opportunity is found, exit the local search for this block
            if not all_opportunities:
                logger.info("수익성 있는 차익거래 기회를 더 이상 찾을 수 없음.")
                break

            # c. Sort opportunities by profitability (net_profit) in descending order
            sorted_opportunities = sorted(all_opportunities, key=lambda op: op.net_profit, reverse=True)
            
            profitable_opportunities_found_in_iteration = False
            for opportunity in sorted_opportunities:
                if opportunity.net_profit > 0.001:
                    profitable_opportunities_found_in_iteration = True
                    # d. Process the opportunity
                    logger.info(f"차익거래 기회 발견: 순수익 {opportunity.net_profit:.6f} ETH")
                    await self._process_opportunities([opportunity])
                    
                    # e. Update the graph state to reflect the executed trade
                    logger.info("그래프 상태를 업데이트합니다.")
                    self.market_graph.update_graph_with_trade(opportunity.edges, opportunity.required_capital)
                else:
                    # Since the list is sorted, we can break if we encounter a non-profitable one
                    break
            
            # f. If no profitable opportunities were found in this iteration, exit
            if not profitable_opportunities_found_in_iteration:
                logger.info("이번 반복에서 수익성 있는 차익거래 기회를 더 이상 찾을 수 없음.")
                break

    async def _update_market_data(self):
        """
        Updates the market graph with the latest pool data from DEXs.
        """
        logger.info(f"{len(self.updated_pools)}개의 풀 데이터 업데이트 시작...")
        dex_configs = {
            'uniswap_v2': {
                'factory': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
                'fee': 0.003
            },
            'sushiswap': { 
                'factory': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac',
                'fee': 0.003
            }
        }

        pools_to_update = list(self.updated_pools)
        self.updated_pools.clear()

        update_tasks = []
        for pool_address in pools_to_update:
            # This is a simplification. We would need a way to know which DEX the pool belongs to.
            # For now, we assume all are Uniswap V2 compatible.
            task = self._update_dex_pool(dex_configs['uniswap_v2'], pool_address)
            update_tasks.append(task)
        
        await asyncio.gather(*update_tasks)
        logger.info("시장 데이터 업데이트 완료.")

    async def _update_dex_pool(self, dex_config: Dict, pool_address: str):
        """
        Fetches and updates the data for a single DEX pool.
        """
        try:
            # Get pair contract and reserves
            pair_contract = self.w3.eth.contract(address=pool_address, abi=self.uniswap_v2_pair_abi)
            reserves = await pair_contract.functions.getReserves().call()
            
            token0_address = await pair_contract.functions.token0().call()
            token1_address = await pair_contract.functions.token1().call()

            token0_info = await self.token_manager.get_token_info(token0_address)
            token1_info = await self.token_manager.get_token_info(token1_address)

            if not token0_info or not token1_info:
                return

            token0_symbol = token0_info.symbol
            token1_symbol = token1_info.symbol

            # Ensure reserves match token order
            reserve0, reserve1 = reserves[0], reserves[1]

            # Get decimals for both tokens
            decimals0 = await self._get_token_decimals(token0_address)
            decimals1 = await self._get_token_decimals(token1_address)
            
            reserve0_adj = reserve0 / (10**decimals0)
            reserve1_adj = reserve1 / (10**decimals1)

            # Update the market graph
            self.market_graph.add_trading_pair(
                token0=token0_address,
                token1=token1_address,
                dex=dex_config['name'],
                pool_address=pool_address,
                reserve0=reserve0_adj,
                reserve1=reserve1_adj,
                fee=dex_config['fee']
            )
        except Exception as e:
            logger.error(f"{dex_config['name']} {pool_address} 풀 업데이트 오류: {e}")

    async def _process_opportunities(self, opportunities: List):
        """Processes found opportunities by logging and storing them."""
        logger.info(f"{len(opportunities)}개의 차익거래 기회 발견 및 처리")
        
        for opp in opportunities:
            logger.info(
                f"차익거래 기회: {' -> '.join(opp.path)} "
                f"수익률: {opp.profit_ratio:.4f} "
                f"순수익: {opp.net_profit:.6f} ETH "
                f"신뢰도: {opp.confidence:.2f}"
            )
            await self.storage.store_arbitrage_opportunity({
                'timestamp': datetime.now().isoformat(),
                'path': opp.path,
                'profit_ratio': opp.profit_ratio,
                'net_profit': opp.net_profit,
                'required_capital': opp.required_capital,
                'confidence': opp.confidence,
                'dexes': [edge.dex for edge in opp.edges]
            })
    
    def stop_detection(self):
        """Stops the detection loop."""
        self.running = False
        logger.info("차익거래 탐지 중지")

async def main():
    """Main execution function."""
    # It's good practice to load environment variables at the start.
    # from dotenv import load_dotenv
    # load_dotenv()
    
    detector = ArbitrageDetector()
    
    try:
        await detector.start_detection()
    except KeyboardInterrupt:
        detector.stop_detection()
        logger.info("프로그램 종료")

if __name__ == "__main__":
    # Add a check for the required environment variable
    if not os.environ.get("ETH_WS_URL"):
        print("오류: ETH_WS_URL 환경 변수가 설정되지 않았습니다.")
        print("예: export ETH_WS_URL=wss://mainnet.infura.io/ws/v3/YOUR_PROJECT_ID")
    else:
        asyncio.run(main())
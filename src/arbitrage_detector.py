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
        self.market_graph = DeFiMarketGraph()
        self.bellman_ford = BellmanFordArbitrage(self.market_graph)
        self.storage = DataStorage()
        self.running = False
        
        # --- Token Configuration ---
        self.token_addresses = {
            "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "USDC": "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        }
        self.base_tokens = list(self.token_addresses.values())
        self.token_decimals = {}

    def _load_abi(self, filepath: str) -> Dict:
        """Helper to load ABI from a file."""
        # Construct path relative to this file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, filepath)) as f:
            return json.load(f)

    async def _get_token_decimals(self, token_address: str) -> int:
        """Fetches and caches the decimals for a given token."""
        if token_address in self.token_decimals:
            return self.token_decimals[token_address]
        
        try:
            token_contract = self.w3.eth.contract(address=token_address, abi=self.erc20_abi)
            decimals = await token_contract.functions.decimals().call()
            self.token_decimals[token_address] = decimals
            logger.info(f"{await token_contract.functions.symbol().call()} ({token_address}) decimals: {decimals}")
            return decimals
        except Exception as e:
            logger.warning(f"토큰 decimals 조회 오류 {token_address}: {e}. 기본값 18 사용.")
            # Fallback to a default value if the call fails
            self.token_decimals[token_address] = 18
            return 18

    async def start_detection(self):
        """
        Starts the arbitrage detection by listening for new blocks and handling them.
        """
        self.running = True
        logger.info("차익거래 탐지 시작 (블록 기반 업데이트)")
        
        # Create a filter for new blocks
        new_block_filter = await self.w3.eth.create_ws_filter('newHeads')
        
        while self.running:
            try:
                for block_hash in await new_block_filter.get_new_entries():
                    logger.info(f"새로운 블록 발견: {block_hash.hex()}")
                    await self._handle_new_block(block_hash)
                
                await asyncio.sleep(1) # Check for new blocks every second

            except Exception as e:
                logger.error(f"블록 리스닝 루프 오류: {e}")
                # In case of a websocket error, try to reconnect
                await asyncio.sleep(10)
                # Re-initialize connection and filter (simplified)
                self.w3 = Web3(Web3.AsyncWebsocketProvider(os.environ.get("ETH_WS_URL", "ws://localhost:8546")))
                new_block_filter = await self.w3.eth.create_ws_filter('newHeads')


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
        Runs the local search loop: find best opportunity, update graph, repeat.
        """
        while True:
            # a. Search for opportunities in parallel across all base tokens
            logger.debug(f"{len(self.base_tokens)}개의 시작점에서 병렬 탐색 시작")
            tasks = [asyncio.to_thread(self.bellman_ford.find_negative_cycles, base_token) for base_token in self.base_tokens]
            results = await asyncio.gather(*tasks)

            all_opportunities = [opp for sublist in results for opp in sublist]
            
            best_opportunity = None
            if all_opportunities:
                best_opportunity = max(all_opportunities, key=lambda op: op.net_profit)

            # b. If no profitable opportunity is found, exit the local search for this block
            if best_opportunity is None or best_opportunity.net_profit <= 0.001:
                logger.info("수익성 있는 차익거래 기회를 더 이상 찾을 수 없음.")
                break
                
            # c. Process the best opportunity found
            logger.info(f"최고의 차익거래 기회 발견: 순수익 {best_opportunity.net_profit:.6f} ETH")
            await self._process_opportunities([best_opportunity])
            
            # d. Update the graph state to reflect the executed trade and repeat search
            logger.info("그래프 상태를 업데이트하고 Local Search를 계속합니다.")
            self.market_graph.update_graph_with_trade(best_opportunity.edges, best_opportunity.required_capital)

    async def _update_market_data(self):
        """
        Updates the market graph with the latest pool data from DEXs.
        """
        logger.info("시장 데이터 업데이트 시작...")
        dex_configs = [
            {
                'name': 'uniswap_v2',
                'factory': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
                'fee': 0.003
            },
            {
                'name': 'sushiswap', 
                'factory': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac',
                'fee': 0.003
            }
        ]
        
        # Create pairs of tokens to check
        token_symbols = list(self.token_addresses.keys())
        major_pairs = []
        for i in range(len(token_symbols)):
            for j in range(i + 1, len(token_symbols)):
                major_pairs.append((token_symbols[i], token_symbols[j]))

        # Update all DEXs and their major pairs
        update_tasks = []
        for dex_config in dex_configs:
            for token0_symbol, token1_symbol in major_pairs:
                task = self._update_dex_pool(dex_config, token0_symbol, token1_symbol)
                update_tasks.append(task)
        
        await asyncio.gather(*update_tasks)
        logger.info("시장 데이터 업데이트 완료.")

    async def _update_dex_pool(self, dex_config: Dict, token0_symbol: str, token1_symbol: str):
        """
        Fetches and updates the data for a single DEX pool.
        """
        try:
            token0_address = self.token_addresses[token0_symbol]
            token1_address = self.token_addresses[token1_symbol]

            # Get factory contract
            factory = self.w3.eth.contract(address=dex_config['factory'], abi=self.uniswap_v2_factory_abi)
            
            # Get pair address
            pair_address = await factory.functions.getPair(token0_address, token1_address).call()

            if pair_address == '0x0000000000000000000000000000000000000000':
                # logger.debug(f"Pair not found for {token0_symbol}/{token1_symbol} on {dex_config['name']}")
                return

            # Get pair contract and reserves
            pair_contract = self.w3.eth.contract(address=pair_address, abi=self.uniswap_v2_pair_abi)
            reserves = await pair_contract.functions.getReserves().call()
            
            # Ensure reserves match token order
            pair_token0 = await pair_contract.functions.token0().call()
            if pair_token0 == token0_address:
                reserve0, reserve1 = reserves[0], reserves[1]
            else:
                reserve0, reserve1 = reserves[1], reserves[0]

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
                pool_address=pair_address,
                reserve0=reserve0_adj,
                reserve1=reserve1_adj,
                fee=dex_config['fee']
            )
        except Exception as e:
            logger.error(f"{dex_config['name']} {token0_symbol}/{token1_symbol} 풀 업데이트 오류: {e}")

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
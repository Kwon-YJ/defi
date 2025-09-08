import asyncio
from web3 import Web3
from web3.contract import Contract
from eth_account import Account
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass
from src.logger import setup_logger
from config.config import config

logger = setup_logger(__name__)

@dataclass
class TradeParams:
    """거래 파라미터"""
    tokens: List[str]
    exchanges: List[str]
    amounts: List[int]
    flash_loan_amount: int
    min_profit: int
    gas_limit: int = 500000
    gas_price: int = 20000000000  # 20 Gwei

class TradeExecutor:
    def __init__(self, web3_provider: Web3, private_key: str):
        self.w3 = web3_provider
        self.account = Account.from_key(private_key)
        self.contract_address = None  # 배포된 컨트랙트 주소
        self.contract = None
        
        # DEX Router 주소들
        self.routers = {
            'uniswap_v2': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
            'sushiswap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
            'curve': '0xd51a44d3fae010294c616388b506acda1bfaae46'
        }
        
        # 토큰 주소들
        self.tokens = {
            'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            'USDC': '0xA0b86a33E6441b8e5c7F5c8b5e8b5e8b5e8b5e8b',
            'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
            'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7'
        }
    
    async def deploy_contract(self) -> str:
        """차익거래 컨트랙트 배포"""
        try:
            # 컨트랙트 ABI와 바이트코드 로드
            with open('abi/FlashArbitrage.json', 'r') as f:
                contract_data = json.load(f)
            
            contract_factory = self.w3.eth.contract(
                abi=contract_data['abi'],
                bytecode=contract_data['bytecode']
            )
            
            # 배포 트랜잭션 생성
            deploy_txn = contract_factory.constructor().build_transaction({
                'from': self.account.address,
                'gas': 2000000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # 트랜잭션 서명 및 전송
            signed_txn = self.account.sign_transaction(deploy_txn)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # 배포 완료 대기
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                self.contract_address = receipt.contractAddress
                self.contract = self.w3.eth.contract(
                    address=self.contract_address,
                    abi=contract_data['abi']
                )
                logger.info(f"컨트랙트 배포 성공: {self.contract_address}")
                return self.contract_address
            else:
                raise Exception("컨트랙트 배포 실패")
                
        except Exception as e:
            logger.error(f"컨트랙트 배포 오류: {e}")
            raise
    
    async def execute_arbitrage(self, opportunity) -> Optional[str]:
        """차익거래 실행"""
        if not self.contract:
            raise Exception("컨트랙트가 배포되지 않았습니다")
        
        try:
            # 거래 파라미터 준비
            trade_params = self._prepare_trade_params(opportunity)
            
            # 가스 추정
            estimated_gas = await self._estimate_gas(trade_params)
            
            # 수익성 재검증
            if not await self._verify_profitability(trade_params, estimated_gas):
                logger.warning("수익성 검증 실패, 거래 취소")
                return None
            
            # 거래 실행
            tx_hash = await self._send_arbitrage_transaction(trade_params, estimated_gas)
            
            if tx_hash:
                logger.info(f"차익거래 거래 전송: {tx_hash.hex()}")
                return tx_hash.hex()
            
            return None
            
        except Exception as e:
            logger.error(f"차익거래 실행 오류: {e}")
            return None
    
    def _prepare_trade_params(self, opportunity) -> TradeParams:
        """거래 파라미터 준비"""
        # 토큰 주소 변환
        token_addresses = []
        for token_symbol in opportunity.path:
            if token_symbol in self.tokens:
                token_addresses.append(self.tokens[token_symbol])
            else:
                token_addresses.append(token_symbol)  # 이미 주소인 경우
        
        # 거래소 주소 변환
        exchange_addresses = []
        for edge in opportunity.edges:
            if edge.dex in self.routers:
                exchange_addresses.append(self.routers[edge.dex])
            else:
                raise ValueError(f"지원하지 않는 DEX: {edge.dex}")
        
        # 거래량 계산
        amounts = self._calculate_trade_amounts(opportunity)
        
        return TradeParams(
            tokens=token_addresses,
            exchanges=exchange_addresses,
            amounts=amounts,
            flash_loan_amount=int(opportunity.required_capital * 1e18),  # Wei 단위
            min_profit=int(opportunity.net_profit * 0.8 * 1e18),  # 80% 안전 마진
            gas_limit=600000,
            gas_price=self.w3.eth.gas_price
        )
    
    def _calculate_trade_amounts(self, opportunity) -> List[int]:
        """각 단계별 거래량 계산"""
        amounts = []
        current_amount = opportunity.required_capital
        
        for edge in opportunity.edges:
            # Wei 단위로 변환
            amount_wei = int(current_amount * 1e18)
            amounts.append(amount_wei)
            
            # 다음 단계 금액 계산 (수수료 고려)
            current_amount = current_amount * edge.exchange_rate * (1 - edge.fee)
        
        return amounts
    
    async def _estimate_gas(self, params: TradeParams) -> int:
        """가스 사용량 추정"""
        try:
            # 컨트랙트 함수 호출로 가스 추정
            gas_estimate = self.contract.functions.executeArbitrage(
                (params.tokens, params.exchanges, params.amounts, 
                 params.flash_loan_amount, params.min_profit)
            ).estimate_gas({'from': self.account.address})
            
            # 안전 마진 추가 (20%)
            return int(gas_estimate * 1.2)
            
        except Exception as e:
            logger.warning(f"가스 추정 실패, 기본값 사용: {e}")
            return params.gas_limit
    
    async def _verify_profitability(self, params: TradeParams, gas_estimate: int) -> bool:
        """수익성 재검증"""
        # 가스 비용 계산
        gas_cost_wei = gas_estimate * params.gas_price
        gas_cost_eth = gas_cost_wei / 1e18
        
        # 예상 순수익 계산
        expected_profit_wei = params.min_profit
        expected_profit_eth = expected_profit_wei / 1e18
        
        net_profit = expected_profit_eth - gas_cost_eth
        
        logger.info(f"수익성 검증: 예상수익 {expected_profit_eth:.6f} ETH, "
                   f"가스비용 {gas_cost_eth:.6f} ETH, 순수익 {net_profit:.6f} ETH")
        
        return net_profit > 0.001  # 최소 0.001 ETH 수익
    
    async def _send_arbitrage_transaction(self, params: TradeParams, gas_limit: int) -> Optional[bytes]:
        """차익거래 트랜잭션 전송"""
        try:
            # 트랜잭션 생성
            txn = self.contract.functions.executeArbitrage(
                (params.tokens, params.exchanges, params.amounts,
                 params.flash_loan_amount, params.min_profit)
            ).build_transaction({
                'from': self.account.address,
                'gas': gas_limit,
                'gasPrice': params.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # 트랜잭션 서명
            signed_txn = self.account.sign_transaction(txn)
            
            # 트랜잭션 전송
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            return tx_hash
            
        except Exception as e:
            logger.error(f"트랜잭션 전송 실패: {e}")
            return None
    
    async def wait_for_confirmation(self, tx_hash: str, timeout: int = 300) -> Dict:
        """트랜잭션 확인 대기"""
        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(
                tx_hash, timeout=timeout
            )
            
            result = {
                'status': 'success' if receipt.status == 1 else 'failed',
                'gas_used': receipt.gasUsed,
                'block_number': receipt.blockNumber,
                'transaction_hash': receipt.transactionHash.hex()
            }
            
            # 이벤트 로그 파싱
            if receipt.status == 1:
                events = self._parse_events(receipt)
                result['events'] = events
                
                # 수익 정보 추출
                for event in events:
                    if event['event'] == 'ArbitrageExecuted':
                        profit_wei = event['args']['profit']
                        profit_eth = profit_wei / 1e18
                        result['profit'] = profit_eth
                        logger.info(f"차익거래 성공! 수익: {profit_eth:.6f} ETH")
            
            return result
            
        except Exception as e:
            logger.error(f"트랜잭션 확인 오류: {e}")
            return {'status': 'timeout', 'error': str(e)}
    
    def _parse_events(self, receipt) -> List[Dict]:
        """이벤트 로그 파싱"""
        events = []
        
        try:
            # ArbitrageExecuted 이벤트
            arbitrage_events = self.contract.events.ArbitrageExecuted().process_receipt(receipt)
            for event in arbitrage_events:
                events.append({
                    'event': 'ArbitrageExecuted',
                    'args': dict(event['args'])
                })
            
            # ArbitrageFailed 이벤트
            failed_events = self.contract.events.ArbitrageFailed().process_receipt(receipt)
            for event in failed_events:
                events.append({
                    'event': 'ArbitrageFailed',
                    'args': dict(event['args'])
                })
                
        except Exception as e:
            logger.warning(f"이벤트 파싱 오류: {e}")
        
        return events

# 시뮬레이션 모드 실행기
class SimulationExecutor:
    """실제 거래 없이 시뮬레이션만 수행"""
    
    def __init__(self, web3_provider: Web3):
        self.w3 = web3_provider
        
    async def simulate_arbitrage(self, opportunity) -> Dict:
        """차익거래 시뮬레이션"""
        try:
            # 각 거래 단계별 시뮬레이션
            simulation_results = []
            current_amount = opportunity.required_capital
            
            for i, edge in enumerate(opportunity.edges):
                # 슬리피지 계산
                slippage = self._calculate_slippage(current_amount, edge.liquidity)
                
                # 실제 받을 수 있는 금액 계산
                output_amount = current_amount * edge.exchange_rate * (1 - slippage)
                
                simulation_results.append({
                    'step': i + 1,
                    'dex': edge.dex,
                    'input_amount': current_amount,
                    'expected_output': current_amount * edge.exchange_rate,
                    'actual_output': output_amount,
                    'slippage': slippage,
                    'fee': edge.fee
                })
                
                current_amount = output_amount
            
            # 최종 수익 계산
            final_profit = current_amount - opportunity.required_capital
            gas_cost = sum(edge.gas_cost for edge in opportunity.edges)
            net_profit = final_profit - gas_cost
            
            return {
                'success': True,
                'steps': simulation_results,
                'initial_capital': opportunity.required_capital,
                'final_amount': current_amount,
                'gross_profit': final_profit,
                'gas_cost': gas_cost,
                'net_profit': net_profit,
                'profit_ratio': current_amount / opportunity.required_capital
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_slippage(self, trade_amount: float, liquidity: float) -> float:
        """슬리피지 계산 (단순 모델)"""
        if liquidity <= 0:
            return 0.1  # 10% 슬리피지 (높은 위험)
        
        # 거래량이 유동성의 몇 %인지에 따라 슬리피지 계산
        impact_ratio = trade_amount / liquidity
        
        if impact_ratio < 0.01:  # 1% 미만
            return 0.001  # 0.1% 슬리피지
        elif impact_ratio < 0.05:  # 5% 미만
            return 0.005  # 0.5% 슬리피지
        elif impact_ratio < 0.1:  # 10% 미만
            return 0.02   # 2% 슬리피지
        else:
            return 0.05   # 5% 슬리피지 (높은 위험)

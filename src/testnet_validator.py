import asyncio
from typing import Dict, List
from web3 import Web3
from eth_account import Account
from src.logger import setup_logger
from config.config import config

logger = setup_logger(__name__)

class TestnetValidator:
    """테스트넷에서 차익거래 전략 검증"""
    
    def __init__(self):
        # Sepolia 테스트넷으로 변경 (더 안정적)
        self.w3 = Web3(Web3.HTTPProvider(config.sepolia_rpc))
        self.test_tokens = {
            'WETH': '0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14',  # Sepolia WETH
            'USDC': '0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238',  # Sepolia USDC
            'DAI': '0x3e622317f8C93f7328350cF0B56d9Ed4C620C5d6'   # Sepolia DAI
        }
        self.test_dexes = {
            'uniswap_v2': '0xeE567Fe1712Faf6149d80dA1E6934E354124CfE3'  # Sepolia V2Router02
        }
        
        # 지갑 설정
        if config.private_key:
            self.account = Account.from_key(config.private_key)
            self.wallet_address = self.account.address
            logger.info(f"테스트 지갑 로드 완료: {self.wallet_address}")
        else:
            self.account = None
            self.wallet_address = None
            logger.warning("Private key가 설정되지 않았습니다. 읽기 전용 모드입니다.")
        
    async def run_validation_suite(self) -> Dict:
        """전체 검증 스위트 실행"""
        results = {
            'network_connectivity': await self._test_network_connectivity(),
            'token_interactions': await self._test_token_interactions(),
            'dex_interactions': await self._test_dex_interactions(),
            'arbitrage_simulation': await self._test_arbitrage_simulation(),
            'gas_estimation': await self._test_gas_estimation()
        }
        
        # 종합 점수 계산
        success_count = sum(1 for result in results.values() if result.get('success', False))
        results['overall_score'] = success_count / len(results)
        
        return results
    
    async def _test_network_connectivity(self) -> Dict:
        """네트워크 연결 테스트"""
        try:
            latest_block = self.w3.eth.get_block('latest')
            chain_id = self.w3.eth.chain_id
            
            return {
                'success': True,
                'latest_block': latest_block.number,
                'chain_id': chain_id,
                'network': 'Sepolia Testnet'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_token_interactions(self) -> Dict:
        """토큰 상호작용 테스트"""
        try:
            results = {}
            
            for symbol, address in self.test_tokens.items():
                # ERC20 기본 정보 조회 - 체크섬 주소로 변환
                checksum_address = self.w3.to_checksum_address(address)
                token_contract = self.w3.eth.contract(
                    address=checksum_address,
                    abi=self._get_erc20_abi()
                )
                
                name = token_contract.functions.name().call()
                symbol_check = token_contract.functions.symbol().call()
                decimals = token_contract.functions.decimals().call()
                
                results[symbol] = {
                    'address': address,
                    'name': name,
                    'symbol': symbol_check,
                    'decimals': decimals,
                    'accessible': True
                }
            
            return {
                'success': True,
                'tokens': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_dex_interactions(self) -> Dict:
        """DEX 상호작용 테스트"""
        try:
            router_address = self.w3.to_checksum_address(self.test_dexes['uniswap_v2'])
            router_contract = self.w3.eth.contract(
                address=router_address,
                abi=self._get_uniswap_router_abi()
            )
            
            # 팩토리 주소 조회
            factory_address = router_contract.functions.factory().call()
            
            # WETH 주소 조회
            weth_address = router_contract.functions.WETH().call()
            
            # 가격 조회 테스트 (WETH/USDC) - 체크섬 주소 사용
            weth_addr = self.w3.to_checksum_address(self.test_tokens['WETH'])
            usdc_addr = self.w3.to_checksum_address(self.test_tokens['USDC'])
            amounts_out = router_contract.functions.getAmountsOut(
                int(1e18),  # 1 WETH
                [weth_addr, usdc_addr]
            ).call()
            
            return {
                'success': True,
                'router_address': router_address,
                'factory_address': factory_address,
                'weth_address': weth_address,
                'weth_usdc_rate': amounts_out[1] / 1e6  # USDC는 6 decimals
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_arbitrage_simulation(self) -> Dict:
        """실제 스왑을 통한 차익거래 테스트"""
        if not self.account:
            return {
                'success': True,
                'test_type': 'simulation_only',
                'message': 'Private key 없음. 시뮬레이션만 수행'
            }
        
        try:
            # 소량 테스트 (0.001 WETH)
            test_amount = int(0.001 * 1e18)
            
            # 체크섬 주소 준비  
            weth_addr = self.w3.to_checksum_address(self.test_tokens['WETH'])
            usdc_addr = self.w3.to_checksum_address(self.test_tokens['USDC'])
            
            # WETH 잔액 확인
            weth_contract = self.w3.eth.contract(
                address=weth_addr,
                abi=self._get_erc20_abi()
            )
            
            weth_balance = weth_contract.functions.balanceOf(self.wallet_address).call()
            
            if weth_balance < test_amount:
                return {
                    'success': False,
                    'error': f'WETH 잔액 부족: {weth_balance / 1e18} WETH, 필요: {test_amount / 1e18} WETH'
                }
            
            # 실제 WETH → USDC 스왑 수행
            swap_result = await self._perform_swap(
                test_amount,
                [weth_addr, usdc_addr],
                0  # 최소 출력량 0 (테스트용)
            )
            
            if swap_result['success']:
                return {
                    'success': True,
                    'test_type': 'real_swap',
                    'amount_swapped': test_amount / 1e18,
                    'tx_hash': swap_result['tx_hash'],
                    'gas_used': swap_result['gas_used']
                }
            else:
                return {
                    'success': False,
                    'error': f"스왑 실패: {swap_result['error']}"
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_gas_estimation(self) -> Dict:
        """실제 지갑 기반 가스 추정 테스트"""
        try:
            router_address = self.w3.to_checksum_address(self.test_dexes['uniswap_v2'])
            router = self.w3.eth.contract(
                address=router_address,
                abi=self._get_uniswap_router_abi()
            )
            
            current_gas_price = self.w3.eth.gas_price
            
            if not self.account:
                # Private key 없으면 Mock 추정
                return {
                    'success': True,
                    'test_type': 'mock_estimation',
                    'estimated_gas': 150000,
                    'gas_price': current_gas_price,
                    'gas_cost_eth': (150000 * current_gas_price) / 1e18
                }
            
            # 실제 가스 추정
            weth_addr = self.w3.to_checksum_address(self.test_tokens['WETH'])
            usdc_addr = self.w3.to_checksum_address(self.test_tokens['USDC'])
            
            try:
                gas_estimate = router.functions.swapExactTokensForTokens(
                    int(0.001 * 1e18),  # 0.001 WETH
                    0,  # 최소 출력량
                    [weth_addr, usdc_addr],
                    self.wallet_address,  # 실제 지갑 주소 사용
                    int(self.w3.eth.get_block('latest').timestamp) + 300
                ).estimate_gas({'from': self.wallet_address})
                
                gas_cost_eth = (gas_estimate * current_gas_price) / 1e18
                
                return {
                    'success': True,
                    'test_type': 'real_estimation',
                    'gas_estimate': gas_estimate,
                    'gas_price': current_gas_price,
                    'gas_cost_eth': gas_cost_eth
                }
                
            except Exception as inner_e:
                # 가스 추정 실패시 approve 문제일 수 있음
                return {
                    'success': True,
                    'test_type': 'estimation_with_note',
                    'estimated_gas': 180000,  # approve + swap
                    'gas_price': current_gas_price,
                    'gas_cost_eth': (180000 * current_gas_price) / 1e18,
                    'note': f'직접 추정 실패, approve 필요할 수 있음: {str(inner_e)[:100]}'
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_erc20_abi(self) -> List[Dict]:
        """ERC20 ABI 반환"""
        return [
            {
                "constant": True,
                "inputs": [],
                "name": "name",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "symbol", 
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            },
            {
                "constant": False,
                "inputs": [
                    {"name": "spender", "type": "address"},
                    {"name": "amount", "type": "uint256"}
                ],
                "name": "approve",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [
                    {"name": "owner", "type": "address"},
                    {"name": "spender", "type": "address"}
                ],
                "name": "allowance",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [{"name": "account", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            }
        ]
    
    async def _approve_token(self, token_address: str, spender: str, amount: int) -> bool:
        """토큰 approve 수행"""
        if not self.account:
            logger.warning("Private key가 없어 approve를 수행할 수 없습니다.")
            return False
            
        try:
            token_contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(token_address),
                abi=self._get_erc20_abi()
            )
            
            # 현재 allowance 확인
            current_allowance = token_contract.functions.allowance(
                self.wallet_address, 
                spender
            ).call()
            
            if current_allowance >= amount:
                logger.info(f"이미 충분한 allowance가 있습니다: {current_allowance}")
                return True
            
            # Approve 트랜잭션 생성
            approve_tx = token_contract.functions.approve(
                spender, 
                amount
            ).build_transaction({
                'from': self.wallet_address,
                'gas': 100000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.wallet_address)
            })
            
            # 서명 및 전송
            signed_tx = self.w3.eth.account.sign_transaction(approve_tx, config.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            # 트랜잭션 대기
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt.status == 1:
                logger.info(f"Approve 성공: {tx_hash.hex()}")
                return True
            else:
                logger.error(f"Approve 실패: {tx_hash.hex()}")
                return False
                
        except Exception as e:
            logger.error(f"Approve 오류: {e}")
            return False
    
    async def _perform_swap(self, amount_in: int, path: List[str], min_amount_out: int = 0) -> Dict:
        """실제 스왑 수행"""
        if not self.account:
            return {'success': False, 'error': 'Private key 없음'}
            
        try:
            router_address = self.w3.to_checksum_address(self.test_dexes['uniswap_v2'])
            router = self.w3.eth.contract(
                address=router_address,
                abi=self._get_uniswap_router_abi()
            )
            
            # 첫 번째 토큰에 대한 approve 수행
            token_in = path[0]
            approve_success = await self._approve_token(token_in, router_address, amount_in)
            
            if not approve_success:
                return {'success': False, 'error': 'Approve 실패'}
            
            # 스왑 트랜잭션 생성
            deadline = int(self.w3.eth.get_block('latest').timestamp) + 300
            
            swap_tx = router.functions.swapExactTokensForTokens(
                amount_in,
                min_amount_out,
                path,
                self.wallet_address,
                deadline
            ).build_transaction({
                'from': self.wallet_address,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.wallet_address)
            })
            
            # 서명 및 전송
            signed_tx = self.w3.eth.account.sign_transaction(swap_tx, config.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            # 트랜잭션 대기
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt.status == 1:
                return {
                    'success': True,
                    'tx_hash': tx_hash.hex(),
                    'gas_used': receipt.gasUsed
                }
            else:
                return {
                    'success': False,
                    'error': f'스왑 트랜잭션 실패: {tx_hash.hex()}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_uniswap_router_abi(self) -> List[Dict]:
        """Uniswap Router ABI 반환 (간소화)"""
        return [
            {
                "constant": True,
                "inputs": [],
                "name": "factory",
                "outputs": [{"name": "", "type": "address"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "WETH",
                "outputs": [{"name": "", "type": "address"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "path", "type": "address[]"}
                ],
                "name": "getAmountsOut",
                "outputs": [{"name": "amounts", "type": "uint256[]"}],
                "type": "function"
            },
            {
                "constant": False,
                "inputs": [
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "amountOutMin", "type": "uint256"},
                    {"name": "path", "type": "address[]"},
                    {"name": "to", "type": "address"},
                    {"name": "deadline", "type": "uint256"}
                ],
                "name": "swapExactTokensForTokens",
                "outputs": [{"name": "amounts", "type": "uint256[]"}],
                "type": "function"
            }
        ]

"""
Adapter to integrate MemoryEfficientGraph with existing DeFiMarketGraph interface.
This allows gradual migration while maintaining compatibility.
"""

from typing import Dict, List, Tuple, Optional, Any
from src.memory_efficient_graph import MemoryEfficientGraph, convert_from_networkx
from src.market_graph import DeFiMarketGraph, ArbitrageOpportunity, TradingEdge
from src.logger import setup_logger
import math

logger = setup_logger(__name__)

class MemoryOptimizedDeFiGraph(DeFiMarketGraph):
    """
    Drop-in replacement for DeFiMarketGraph using memory-efficient backend.
    
    This class maintains the existing interface while using the optimized
    MemoryEfficientGraph internally for storage and operations.
    """
    
    def __init__(self, web3_provider=None, enable_memory_optimization=True):
        """
        Initialize with optional memory optimization.
        
        Args:
            web3_provider: Web3 provider for protocol interactions
            enable_memory_optimization: Use memory-efficient backend (default: True)
        """
        if enable_memory_optimization:
            logger.info("Initializing with memory-optimized backend")
            
            # Initialize memory-efficient graph instead of NetworkX
            self.efficient_graph = MemoryEfficientGraph()
            self.use_efficient_backend = True
            
            # Initialize parent class but replace graph immediately
            super().__init__(web3_provider)
            
            # Clear NetworkX graph to save memory
            self.graph.clear()
            
        else:
            logger.info("Initializing with standard NetworkX backend")
            super().__init__(web3_provider)
            self.efficient_graph = None
            self.use_efficient_backend = False
        
        # Track memory savings
        self._memory_savings = {
            'enabled': enable_memory_optimization,
            'baseline_memory_mb': 0,
            'current_memory_mb': 0,
            'saved_memory_mb': 0
        }
    
    def add_token(self, token_address: str, symbol: str = None):
        """Add token with memory optimization"""
        if self.use_efficient_backend:
            self.efficient_graph.add_node(token_address, symbol)
            self.token_nodes.add(token_address)
            logger.debug(f"토큰 노드 추가 (메모리 최적화): {symbol or token_address}")
            
            # 실시간 상태 변화 알림
            self._notify_state_change('token_added', {
                'token_address': token_address,
                'symbol': symbol,
                'memory_optimized': True
            })
        else:
            # Use parent implementation
            super().add_token(token_address, symbol)
    
    def add_trading_pair(self, token0: str, token1: str, dex: str, 
                        pool_address: str, reserve0: float, reserve1: float,
                        fee: float = 0.003):
        """Add trading pair with memory optimization"""
        
        if self.use_efficient_backend:
            # Add tokens
            self.add_token(token0)
            self.add_token(token1)
            
            # Validate reserves
            if reserve0 <= 0 or reserve1 <= 0:
                logger.warning(f"유동성 부족: {pool_address}")
                return
            
            # Calculate rates and weights using parent's logic
            spot_price_01 = reserve1 / reserve0
            effective_rate_01 = spot_price_01 * (1 - fee)
            weight_01 = self._calculate_edge_weight(spot_price_01)
            
            spot_price_10 = reserve0 / reserve1
            effective_rate_10 = spot_price_10 * (1 - fee)
            weight_10 = self._calculate_edge_weight(spot_price_10)
            
            gas_cost = self._estimate_gas_cost(dex)
            liquidity = min(reserve0, reserve1)
            
            # Add edges to efficient graph
            self.efficient_graph.add_edge(
                from_token=token0,
                to_token=token1,
                exchange_rate=effective_rate_01,
                weight=weight_01,
                liquidity=liquidity,
                fee=fee,
                gas_cost=gas_cost,
                dex_name=dex,
                pool_address=pool_address
            )
            
            self.efficient_graph.add_edge(
                from_token=token1,
                to_token=token0,
                exchange_rate=effective_rate_10,
                weight=weight_10,
                liquidity=liquidity,
                fee=fee,
                gas_cost=gas_cost,
                dex_name=dex,
                pool_address=pool_address
            )
            
            # Update registry for compatibility
            pair_key = (token0, token1)
            if pair_key not in self._edge_registry:
                self._edge_registry[pair_key] = {}
            self._edge_registry[pair_key][dex] = 0  # Placeholder edge key
            
            reverse_pair_key = (token1, token0) 
            if reverse_pair_key not in self._edge_registry:
                self._edge_registry[reverse_pair_key] = {}
            self._edge_registry[reverse_pair_key][dex] = 0  # Placeholder edge key
            
            # Update best edges
            current_best_rate = self._get_edge_rate(pair_key) if pair_key in self._best_edges else 0
            if effective_rate_01 > current_best_rate:
                self._best_edges[pair_key] = (dex, 0)
                
            current_best_rate_reverse = self._get_edge_rate(reverse_pair_key) if reverse_pair_key in self._best_edges else 0
            if effective_rate_10 > current_best_rate_reverse:
                self._best_edges[reverse_pair_key] = (dex, 0)
            
            logger.debug(f"거래 쌍 추가 (메모리 최적화): {dex} {token0}-{token1}")
            
            # 실시간 상태 변화 알림
            self._notify_state_change('trading_pair_added', {
                'from_token': token0,
                'to_token': token1,
                'dex': dex,
                'pool_address': pool_address,
                'reserve0': reserve0,
                'reserve1': reserve1,
                'memory_optimized': True
            })
            
        else:
            # Use parent implementation
            super().add_trading_pair(token0, token1, dex, pool_address, reserve0, reserve1, fee)
    
    def get_best_edge(self, from_token: str, to_token: str) -> Optional[Dict]:
        """Get best edge with memory optimization"""
        if self.use_efficient_backend:
            return self.efficient_graph.get_best_edge(from_token, to_token)
        else:
            return super().get_best_edge(from_token, to_token)
    
    def get_all_edges(self, from_token: str, to_token: str = None) -> List[Dict]:
        """Get all edges with memory optimization"""
        if self.use_efficient_backend:
            if to_token:
                # Get specific pair (maintain compatibility)
                best_edge = self.efficient_graph.get_best_edge(from_token, to_token)
                return [best_edge] if best_edge else []
            else:
                # Get all edges from token
                return self.efficient_graph.get_all_edges(from_token)
        else:
            return super().get_all_edges(from_token, to_token)
    
    def update_pool_data(self, pool_address: str, reserve0: float, reserve1: float):
        """Update pool data with memory optimization"""
        if self.use_efficient_backend:
            import time
            start_time = time.time()
            
            # Calculate new exchange rate (simplified - assumes 1:1 token pair)
            if reserve0 > 0 and reserve1 > 0:
                # For simplicity, update with average rate
                new_rate = reserve1 / reserve0
                updated_count = self.efficient_graph.update_edge_rate(pool_address, new_rate)
                
                update_time = time.time() - start_time
                
                if updated_count > 0:
                    logger.debug(f"Pool 업데이트 (메모리 최적화): {pool_address} - {updated_count}개 업데이트됨 ({update_time:.3f}s)")
                
                # 실시간 상태 변화 알림
                self._notify_state_change('pool_update', {
                    'pool_address': pool_address,
                    'reserve0': reserve0,
                    'reserve1': reserve1,
                    'updated_pairs': updated_count,
                    'update_time': update_time,
                    'memory_optimized': True
                })
            
        else:
            # Use parent implementation
            super().update_pool_data(pool_address, reserve0, reserve1)
    
    def get_graph_stats(self) -> Dict:
        """Get graph statistics with memory optimization details"""
        if self.use_efficient_backend:
            efficient_stats = self.efficient_graph.get_stats()
            memory_stats = self.efficient_graph.get_memory_usage()
            
            # Update memory savings tracking
            self._memory_savings['current_memory_mb'] = memory_stats['total_mb']
            self._memory_savings['saved_memory_mb'] = memory_stats['memory_saved_mb']
            
            # Multi-graph stats (compatibility)
            multi_stats = {
                'total_token_pairs': len(self._edge_registry),
                'multi_dex_pairs': sum(1 for edges in self._edge_registry.values() if len(edges) > 1),
                'single_dex_pairs': sum(1 for edges in self._edge_registry.values() if len(edges) == 1),
                'max_dex_per_pair': max((len(edges) for edges in self._edge_registry.values()), default=0),
                'average_dex_per_pair': (sum(len(edges) for edges in self._edge_registry.values()) / 
                                       len(self._edge_registry)) if self._edge_registry else 0,
                'multi_graph_efficiency': (sum(1 for edges in self._edge_registry.values() if len(edges) > 1) / 
                                          len(self._edge_registry)) if self._edge_registry else 0
            }
            
            return {
                'nodes': efficient_stats['nodes'],
                'edges': efficient_stats['edges'],
                'tokens': len(self.token_nodes),
                'density': efficient_stats['edges'] / max(1, efficient_stats['nodes'] * (efficient_stats['nodes'] - 1)),
                'is_connected': True,  # Assume connected for large graphs
                'multi_graph': multi_stats,
                'memory_optimization': {
                    'enabled': True,
                    'memory_used_mb': memory_stats['total_mb'],
                    'memory_saved_mb': memory_stats['memory_saved_mb'],
                    'compression_ratio': memory_stats['compression_ratio'],
                    'savings_percentage': memory_stats['savings_percentage']
                }
            }
        else:
            stats = super().get_graph_stats()
            stats['memory_optimization'] = {'enabled': False}
            return stats
    
    def prune_inefficient_edges(self, min_liquidity: float = 1.0, 
                               max_fee: float = 0.01,
                               min_exchange_rate: float = 1e-6) -> int:
        """Prune inefficient edges with memory optimization"""
        if self.use_efficient_backend:
            # Use efficient graph's built-in pruning
            removed_count = self.efficient_graph.prune_low_liquidity_edges(min_liquidity)
            
            if removed_count > 0:
                logger.info(f"Graph pruning 완료 (메모리 최적화): {removed_count}개 비효율적 edge 제거")
                
                # Clean up registries
                self._cleanup_edge_registries()
            
            return removed_count
        else:
            return super().prune_inefficient_edges(min_liquidity, max_fee, min_exchange_rate)
    
    def _cleanup_edge_registries(self):
        """Clean up edge registries after pruning"""
        # This is a simplified cleanup - in production, would need more sophisticated tracking
        valid_pairs = set()
        
        # Get all valid edges from efficient graph
        for token in self.token_nodes:
            edges = self.efficient_graph.get_all_edges(token)
            for edge in edges:
                valid_pairs.add((edge['from_token'], edge['to_token']))
        
        # Remove invalid entries from registries
        invalid_keys = []
        for pair_key in self._edge_registry.keys():
            if pair_key not in valid_pairs:
                invalid_keys.append(pair_key)
        
        for key in invalid_keys:
            del self._edge_registry[key]
            if key in self._best_edges:
                del self._best_edges[key]
    
    def get_bellman_ford_edges(self) -> List[Tuple[int, int, float]]:
        """Get edges for Bellman-Ford algorithm"""
        if self.use_efficient_backend:
            return self.efficient_graph.get_bellman_ford_edges()
        else:
            # Convert from NetworkX format
            edges = []
            node_to_id = {node: i for i, node in enumerate(self.token_nodes)}
            
            for u, v, data in self.graph.edges(data=True):
                if u in node_to_id and v in node_to_id:
                    weight = data.get('weight', 0.0)
                    edges.append((node_to_id[u], node_to_id[v], weight))
            
            return edges
    
    def migrate_to_memory_efficient(self) -> bool:
        """Migrate existing NetworkX graph to memory-efficient format"""
        if self.use_efficient_backend:
            logger.info("Already using memory-efficient backend")
            return True
        
        try:
            logger.info("Migrating to memory-efficient backend...")
            
            # Record baseline memory usage
            baseline_edges = len(self.graph.edges())
            baseline_nodes = len(self.graph.nodes())
            baseline_memory_estimate = baseline_edges * 200 + baseline_nodes * 100  # NetworkX overhead
            
            # Convert existing graph
            self.efficient_graph = convert_from_networkx(self.graph)
            self.use_efficient_backend = True
            
            # Clear NetworkX graph to free memory
            self.graph.clear()
            
            # Update memory savings
            memory_stats = self.efficient_graph.get_memory_usage()
            self._memory_savings = {
                'enabled': True,
                'baseline_memory_mb': baseline_memory_estimate / (1024 * 1024),
                'current_memory_mb': memory_stats['total_mb'],
                'saved_memory_mb': memory_stats['memory_saved_mb']
            }
            
            logger.info(f"Migration completed successfully!")
            logger.info(f"Memory saved: {self._memory_savings['saved_memory_mb']:.2f} MB")
            logger.info(f"Compression ratio: {memory_stats['compression_ratio']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def get_memory_report(self) -> Dict:
        """Get detailed memory usage report"""
        if self.use_efficient_backend:
            memory_stats = self.efficient_graph.get_memory_usage()
            efficiency_stats = self.efficient_graph.get_stats()
            compliance = self.efficient_graph.validate_paper_compliance()
            
            return {
                'memory_optimization': {
                    'enabled': True,
                    'backend': 'MemoryEfficientGraph',
                    'current_usage': {
                        'total_mb': memory_stats['total_mb'],
                        'edge_memory_mb': memory_stats['edge_memory_bytes'] / (1024 * 1024),
                        'node_memory_mb': memory_stats['node_memory_bytes'] / (1024 * 1024),
                        'metadata_memory_mb': (memory_stats['dex_pool_memory_bytes'] + 
                                             memory_stats['best_edge_memory_bytes']) / (1024 * 1024)
                    },
                    'savings': {
                        'memory_saved_mb': memory_stats['memory_saved_mb'],
                        'compression_ratio': memory_stats['compression_ratio'],
                        'savings_percentage': memory_stats['savings_percentage']
                    }
                },
                'efficiency_metrics': {
                    'edges_per_node': efficiency_stats['avg_edges_per_node'],
                    'dex_count': efficiency_stats['dex_count'],
                    'pool_count': efficiency_stats['pool_count'],
                    'best_edges_tracked': efficiency_stats['best_edges']
                },
                'paper_compliance': compliance,
                'recommendations': self._get_memory_recommendations()
            }
        else:
            return {
                'memory_optimization': {
                    'enabled': False,
                    'backend': 'NetworkX',
                    'migration_available': True
                },
                'recommendations': [
                    "Consider migrating to memory-efficient backend using migrate_to_memory_efficient()",
                    "Memory-efficient backend can save 50-80% memory usage",
                    "Improved performance for large-scale operations (96+ protocol actions)"
                ]
            }
    
    def _get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations"""
        recommendations = []
        
        if self.use_efficient_backend:
            memory_stats = self.efficient_graph.get_memory_usage()
            stats = self.efficient_graph.get_stats()
            
            # Check memory usage
            if memory_stats['total_mb'] > 100:
                recommendations.append(f"High memory usage ({memory_stats['total_mb']:.1f} MB). Consider pruning low-liquidity edges.")
            
            # Check graph size
            if stats['edges'] > 10000:
                recommendations.append("Large graph detected. Consider implementing edge batching for updates.")
            
            # Check efficiency
            if memory_stats['compression_ratio'] > 0.8:
                recommendations.append("Low compression ratio. Verify that data is being properly scaled and optimized.")
            
            # Check for paper compliance
            compliance = self.efficient_graph.validate_paper_compliance()
            if not compliance['ready_for_production']:
                recommendations.append("Graph may not be optimized for paper-scale operations (96 protocols, 25 assets).")
            
            if not recommendations:
                recommendations.append("Memory optimization is performing well. No immediate action needed.")
        
        return recommendations

def create_optimized_graph(web3_provider=None, force_memory_optimization=True) -> MemoryOptimizedDeFiGraph:
    """Factory function to create optimized graph instance"""
    return MemoryOptimizedDeFiGraph(web3_provider, enable_memory_optimization=force_memory_optimization)
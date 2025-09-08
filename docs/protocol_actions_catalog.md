# Protocol Actions Catalog (Working Draft)

Note: This is a working catalog aiming for 96 protocol actions aligned with the paper’s spirit. Due to offline constraints, exact one-to-one matching to the paper’s list requires later validation. This draft provides a concrete target set and mapping into our ActionRegistry for progressive implementation.

DEX (Layer 1 AMMs)
1. Uniswap V2: swapExactTokensForTokens
2. Uniswap V2: swapTokensForExactTokens
3. Uniswap V2: swapExactETHForTokens
4. Uniswap V2: swapTokensForExactETH
5. Uniswap V2: swapExactTokensForETH
6. Uniswap V2: swapETHForExactTokens
7. Uniswap V2: addLiquidity
8. Uniswap V2: addLiquidityETH
9. Uniswap V2: removeLiquidity
10. Uniswap V2: removeLiquidityETH

11. SushiSwap (V2): swapExactTokensForTokens
12. SushiSwap (V2): swapTokensForExactTokens
13. SushiSwap (V2): swapExactETHForTokens
14. SushiSwap (V2): swapTokensForExactETH
15. SushiSwap (V2): swapExactTokensForETH
16. SushiSwap (V2): swapETHForExactTokens
17. SushiSwap (V2): addLiquidity
18. SushiSwap (V2): addLiquidityETH
19. SushiSwap (V2): removeLiquidity
20. SushiSwap (V2): removeLiquidityETH

21. Uniswap V3: exactInputSingle
22. Uniswap V3: exactInput
23. Uniswap V3: exactOutputSingle
24. Uniswap V3: exactOutput
25. Uniswap V3: addLiquidity (mint)
26. Uniswap V3: decreaseLiquidity
27. Uniswap V3: collect
28. Uniswap V3: multicall route

Curve (Stable AMMs)
29. Curve: exchange(i,j)
30. Curve: add_liquidity
31. Curve: remove_liquidity
32. Curve: remove_liquidity_imbalance
33. Curve MetaPool: exchange(i,j)
34. Curve MetaPool: add_liquidity
35. Curve MetaPool: remove_liquidity

Balancer
36. Balancer V1: swapExactIn
37. Balancer V1: swapExactOut
38. Balancer V1: joinPool
39. Balancer V1: exitPool
40. Balancer V2: batchSwap
41. Balancer V2: joinPool
42. Balancer V2: exitPool

Lending / Borrowing
43. Aave V2: deposit (supply)
44. Aave V2: withdraw
45. Aave V2: borrow
46. Aave V2: repay
47. Aave V2: flashLoan
48. Aave V3: supply
49. Aave V3: withdraw
50. Aave V3: borrow
51. Aave V3: repay
52. Aave V3: flashLoan

53. Compound V2: mint (supply)
54. Compound V2: redeem
55. Compound V2: borrow
56. Compound V2: repayBorrow
57. Compound V2: repayBorrowBehalf
58. Compound V2: liquidateBorrow
59. Compound V3: supply
60. Compound V3: withdraw
61. Compound V3: borrow
62. Compound V3: repay
63. Compound V3: liquidate

CDP / Vaults / Yield
64. MakerDAO: open (open vault)
65. MakerDAO: lock (collateral)
66. MakerDAO: draw (mint DAI)
67. MakerDAO: wipe (repay DAI)
68. MakerDAO: free (unlock collateral)
69. MakerDAO: shut (close vault)

70. Yearn: deposit (vault)
71. Yearn: withdraw (vault)
72. Yearn: claim (rewards)

Synthetics / Derivatives
73. Synthetix: exchange
74. Synthetix: mint
75. Synthetix: burn
76. Synthetix: claimRewards
77. dYdX: deposit
78. dYdX: withdraw
79. dYdX: transfer
80. dYdX: trade
81. dYdX: liquidate

Aggregators / Other AMMs
82. Kyber: swapExactTokensForTokens
83. Kyber: swapTokensForExactTokens
84. 1inch: swap (direct route)
85. ParaSwap: swap (direct route)
86. Bancor: swap
87. Bancor: addLiquidity
88. Bancor: removeLiquidity
89. mStable: mint (mAsset)
90. mStable: redeem
91. mStable: swap

Perps / Other
92. GMX: open position
93. GMX: close position
94. GMX: swap

Routers / Utilities
95. Uniswap Universal Router: execute
96. Flash loan aggregator: execute route (generic)

This catalog is intended for progressive implementation via the ActionRegistry. Each action corresponds to a concrete interaction template that can be modeled into the market graph (edges) and, where applicable, executed in simulation or on-chain via dedicated executors.


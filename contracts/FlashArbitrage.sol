// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

interface IUniswapV2Router {
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
    
    function getAmountsOut(uint amountIn, address[] calldata path)
        external view returns (uint[] memory amounts);
}

interface IFlashLoanProvider {
    function flashLoan(
        address asset,
        uint256 amount,
        bytes calldata params
    ) external;
}

interface IFlashLoanReceiver {
    function executeFlashLoan(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external returns (bool);
}

contract FlashArbitrage is ReentrancyGuard, Ownable, IFlashLoanReceiver {
    
    struct ArbitrageParams {
        address[] tokens;           // 거래 경로의 토큰들
        address[] exchanges;        // 사용할 거래소들
        uint256[] amounts;          // 각 단계별 거래량
        uint256 flashLoanAmount;    // 플래시 론 금액
        uint256 minProfit;          // 최소 수익
    }
    
    mapping(address => bool) public authorizedCallers;
    mapping(address => IUniswapV2Router) public routers;
    
    event ArbitrageExecuted(
        address indexed token,
        uint256 flashLoanAmount,
        uint256 profit,
        address indexed executor
    );
    
    event ArbitrageFailed(
        address indexed token,
        uint256 flashLoanAmount,
        string reason
    );
    
    modifier onlyAuthorized() {
        require(authorizedCallers[msg.sender], "Not authorized");
        _;
    }
    
    constructor() {
        // Uniswap V2 Router
        routers[0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D] = 
            IUniswapV2Router(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        
        // SushiSwap Router  
        routers[0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F] = 
            IUniswapV2Router(0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F);
    }
    
    function executeArbitrage(ArbitrageParams calldata params) 
        external 
        onlyAuthorized 
        nonReentrant 
    {
        require(params.tokens.length >= 2, "Invalid token path");
        require(params.exchanges.length == params.tokens.length - 1, "Mismatched arrays");
        
        // 플래시 론 요청
        address flashLoanProvider = 0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9; // Aave V2
        bytes memory data = abi.encode(params);
        
        IFlashLoanProvider(flashLoanProvider).flashLoan(
            params.tokens[0],
            params.flashLoanAmount,
            data
        );
    }
    
    function executeFlashLoan(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        
        ArbitrageParams memory arbParams = abi.decode(params, (ArbitrageParams));
        
        uint256 initialBalance = IERC20(asset).balanceOf(address(this));
        uint256 currentAmount = amount;
        
        try this.performArbitrageSteps(arbParams, currentAmount) returns (uint256 finalAmount) {
            
            uint256 totalDebt = amount + premium;
            require(finalAmount >= totalDebt + arbParams.minProfit, "Insufficient profit");
            
            // 플래시 론 상환
            IERC20(asset).transfer(msg.sender, totalDebt);
            
            // 수익 계산 및 이벤트 발생
            uint256 profit = finalAmount - totalDebt;
            emit ArbitrageExecuted(asset, amount, profit, initiator);
            
            return true;
            
        } catch Error(string memory reason) {
            emit ArbitrageFailed(asset, amount, reason);
            revert(reason);
        }
    }
    
    function performArbitrageSteps(ArbitrageParams memory params, uint256 startAmount) 
        external 
        returns (uint256) 
    {
        require(msg.sender == address(this), "Internal function");
        
        uint256 currentAmount = startAmount;
        
        for (uint256 i = 0; i < params.exchanges.length; i++) {
            address tokenIn = params.tokens[i];
            address tokenOut = params.tokens[i + 1];
            address exchange = params.exchanges[i];
            
            // 토큰 승인
            IERC20(tokenIn).approve(exchange, currentAmount);
            
            // 거래 실행
            currentAmount = _executeSwap(
                exchange,
                tokenIn,
                tokenOut,
                currentAmount,
                0  // 최소 출력량은 별도 계산
            );
        }
        
        return currentAmount;
    }
    
    function _executeSwap(
        address router,
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 amountOutMin
    ) internal returns (uint256) {
        
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;
        
        uint256[] memory amounts = IUniswapV2Router(router).swapExactTokensForTokens(
            amountIn,
            amountOutMin,
            path,
            address(this),
            block.timestamp + 300
        );
        
        return amounts[amounts.length - 1];
    }
    
    // 관리자 함수들
    function addAuthorizedCaller(address caller) external onlyOwner {
        authorizedCallers[caller] = true;
    }
    
    function removeAuthorizedCaller(address caller) external onlyOwner {
        authorizedCallers[caller] = false;
    }
    
    function emergencyWithdraw(address token) external onlyOwner {
        uint256 balance = IERC20(token).balanceOf(address(this));
        IERC20(token).transfer(owner(), balance);
    }
}

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
}

// Aave V3 simple flash loan interfaces
interface IPool {
    function flashLoanSimple(
        address receiverAddress,
        address asset,
        uint256 amount,
        bytes calldata params,
        uint16 referralCode
    ) external;
}

interface IFlashLoanSimpleReceiver {
    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external returns (bool);
}

contract FlashArbitrageAaveV3 is ReentrancyGuard, Ownable, IFlashLoanSimpleReceiver {
    struct ArbitrageParams {
        address[] tokens;           // 경로 토큰들 (대여 자산 = tokens[0])
        address[] exchanges;        // 사용 DEX 라우터들
        uint256[] amounts;          // 각 단계 입력 금액(옵션)
        uint256 flashLoanAmount;    // 플래시 대여 금액
        uint256 minProfit;          // 최소 수익 (대여 자산 기준)
    }

    IPool public pool; // Aave V3 Pool
    mapping(address => bool) public authorizedCallers;
    mapping(address => IUniswapV2Router) public routers; // UniswapV2 호환 라우터

    event ArbitrageExecuted(address indexed asset, uint256 amount, uint256 profit, address indexed executor);
    event ArbitrageFailed(address indexed asset, uint256 amount, string reason);

    modifier onlyAuthorized() {
        require(authorizedCallers[msg.sender], "Not authorized");
        _;
    }

    constructor(address poolAddress) {
        pool = IPool(poolAddress);
        // Uniswap V2
        routers[0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D] = IUniswapV2Router(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        // SushiSwap
        routers[0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F] = IUniswapV2Router(0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F);
    }

    function startArbitrage(ArbitrageParams calldata params) external onlyAuthorized nonReentrant {
        require(params.tokens.length >= 2, "invalid path");
        require(params.exchanges.length == params.tokens.length - 1, "mismatch arrays");
        bytes memory data = abi.encode(params);
        pool.flashLoanSimple(address(this), params.tokens[0], params.flashLoanAmount, data, 0);
    }

    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        ArbitrageParams memory p = abi.decode(params, (ArbitrageParams));

        uint256 currentAmount = amount;
        for (uint256 i = 0; i < p.exchanges.length; i++) {
            address tokenIn = p.tokens[i];
            address tokenOut = p.tokens[i + 1];
            address router = p.exchanges[i];

            // 승인 및 스왑 수행
            IERC20(tokenIn).approve(router, currentAmount);
            address[] memory path = new address[](2);
            path[0] = tokenIn;
            path[1] = tokenOut;
            uint[] memory amountsOut = IUniswapV2Router(router).swapExactTokensForTokens(
                currentAmount, 0, path, address(this), block.timestamp + 300
            );
            currentAmount = amountsOut[amountsOut.length - 1];
        }

        uint256 debt = amount + premium;
        require(currentAmount >= debt + p.minProfit, "Insufficient profit");

        // 상환 승인 및 실행자 이벤트
        IERC20(asset).approve(address(pool), debt);
        emit ArbitrageExecuted(asset, amount, currentAmount - debt, initiator);
        return true;
    }

    // Admin
    function addAuthorizedCaller(address caller) external onlyOwner { authorizedCallers[caller] = true; }
    function removeAuthorizedCaller(address caller) external onlyOwner { authorizedCallers[caller] = false; }
    function emergencyWithdraw(address token) external onlyOwner {
        uint256 bal = IERC20(token).balanceOf(address(this));
        IERC20(token).transfer(owner(), bal);
    }
}


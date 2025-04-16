# Deep Q-Learning Portfolio Trading Strategy

## Overview

This repository contains an implementation of an intelligent portfolio trading strategy using Deep Q-Learning, based on the methodology presented in the research paper "An intelligent financial portfolio trading strategy using deep Q-learning" by Hyungjun Park, Min Kyu Sim, and Dong Gu Choi.

The code implements a reinforcement learning agent that learns to make optimal trading decisions for a portfolio of multiple assets, aiming to maximize returns while managing risk and transaction costs. Unlike many existing approaches, this implementation:

1. Uses a discrete combinatorial action space for practical applicability
2. Implements a mapping function to handle infeasible actions
3. Overcomes dimensionality problems to enable multi-asset trading
4. Employs a simulation technique to derive well-fitted strategies

## Code Structure

The implementation consists of several key components:

### 1. Portfolio Environment (`PortfolioEnv` class)

This class defines the trading environment where the agent interacts by taking actions and receiving rewards. It includes:

- Market data processing and technical indicator calculation
- Portfolio state tracking
- Action space definition and management
- Reward calculation
- Handling of portfolio evolution through time

### 2. Prioritized Experience Replay

Implements two helper classes:
- `SumTree`: An efficient priority storage structure
- `PrioritizedReplayBuffer`: Stores experiences with priorities for more efficient learning

### 3. Metrics Tracking (`MetricsTracker` class)

Tracks and calculates various training metrics including:
- Portfolio returns
- Action distributions
- Q-values
- Loss values
- Transaction costs

### 4. Deep Q-Network Model

A neural network architecture with:
- Shared LSTM encoders for technical indicators
- Custom transpose layer
- Multiple dense layers with normalization and regularization
- Q-value output for each possible action

### 5. Training Functions

Key functions for the training process:
- `train_step`: TensorFlow function for model training
- `prepare_inputs_from_state`: Prepares environment state for model input
- `evaluate_model`: Evaluates model performance on test data

### 6. Visualization

Functions for visualizing and analyzing:
- Portfolio weight evolution
- Portfolio value growth
- Comparison with benchmark strategies (Buy & Hold)

## Methodology

### Portfolio Formulation

The portfolio consists of cash and multiple risky assets (ETFs or indices). The agent makes trading decisions based on:

1. Technical indicators calculated from historical price data
2. Current portfolio weights

### State Space

The state representation includes:
- Technical indicators for each asset (close, open, high, low, volume)
- Current portfolio weights

Each technical indicator is processed over a time window of size `n` (default: 20 days) to capture temporal patterns.

### Action Space

The action space consists of discrete trading decisions for each asset:
- Sell (action = 0): Sell a fixed amount of the asset
- Hold (action = 1): Make no changes to asset position
- Buy (action = 2): Buy a fixed amount of the asset

Each possible combination of actions for all assets is considered, resulting in a discrete combinatorial action space.

### Algorithm Features

1. **Mapping Function**: When the agent selects an infeasible action (due to cash shortage or asset shortage), a mapping function transforms it into a similar feasible action.

2. **State Dimension Reduction**: The LSTM encoder reduces the dimensions of technical indicators to make the problem tractable.

3. **Simulating All Feasible Actions**: For each state, all feasible actions are simulated in parallel, allowing the agent to learn efficiently from multiple experiences.

4. **Dynamic Epsilon**: Exploration rate (epsilon) adjusts based on market volatility, enabling adaptive exploration.

## Model Architecture

The Deep Q-Network consists of:

1. **Encoder**: Shared LSTM layers that process technical indicators for each asset
2. **Feature Combination**: Concatenation of encoded asset features with portfolio weights
3. **Decision Network**: Dense layers with regularization and normalization
4. **Output Layer**: Q-values for each possible action

## How to Use the Code

### Environment Setup

# Define dataset and parameters
datasets = ["SPY", "IWD", "IWC"]  # Assets in portfolio
n = 20  # Time window size

# Create training environment
train_dates = ["2010-01-01", "2010-12-30"]
env = PortfolioEnv(train_dates, datasets, n)


### Model Training

# Build model
model = build_model(datasets, env.action_shape)

# Train the model
# See main training loop in the code for full implementation


### Model Evaluation


# Evaluate model on test data
test_dates = ["2016-01-01", "2016-12-30"]
test_env = PortfolioEnv(test_dates, datasets, n)
results = evaluate_model(model, test_env)


### Visualizing Results


# Visualize portfolio evolution
portfolio_data = run_and_plot_portfolio_evolution(model, test_env)

# Compare with Buy & Hold strategy
bh_values = calculate_buy_and_hold(test_env)
plot_portfolio_evolution_with_bh(portfolio_data, bh_values)


## Key Parameters

- `datasets`: List of asset symbols in the portfolio
- `n`: Size of time window for technical indicators
- `BATCH_SIZE`: Number of experiences used in each training step
- `DISCOUNT_RATE`: Factor for future reward discounting
- `c_minus`, `c_plus`: Transaction costs for selling and buying
- `delta`: Fixed trading size for each action

## Performance Metrics

The strategy is evaluated using several metrics:

- **Portfolio Return**: Percentage return over the investment period
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sterling Ratio**: Downside risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Action Distribution**: Frequency of buy, sell, and hold actions

## Implementation Details

### Technical Indicators

Five technical indicators are used for each asset:
1. Close price change rate
2. Open-to-previous-close ratio
3. Close-to-high ratio
4. Close-to-low ratio
5. Volume change rate

### Training Process

1. The agent interacts with the environment by selecting actions
2. All feasible actions are simulated to generate experiences
3. Experiences are stored in prioritized replay memory
4. Batches of experiences are sampled for model training
5. The model is updated using the Huber loss function
6. Target network is periodically updated

### Testing Process

During testing:
1. The trained agent selects actions with the highest Q-values
2. Infeasible actions are mapped to feasible ones
3. Performance metrics are tracked and calculated
4. Results are compared with benchmark strategies

## Advantages

This implementation offers several advantages over traditional portfolio management approaches:

1. **Adaptive Learning**: The agent adapts to changing market conditions
2. **Practical Action Space**: Trading decisions are directly applicable
3. **Transaction Cost Awareness**: The strategy considers transaction costs
4. **Multi-Asset Trading**: Can handle multiple assets simultaneously
5. **Controlled Turnover**: Fixed trading size helps control portfolio turnover

## References
Park, H., Sim, M. K., & Choi, D. G. (2020). An intelligent financial portfolio trading strategy using deep Q-learning. Expert Systems with Applications, 158, 113573.


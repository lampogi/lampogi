# Genetic Algorithm Trading Strategy Optimizer

A Streamlit research dashboard based on the uploaded thesis, **Optimizing Technical Trading Strategies Using Genetic Algorithms and Risk-Adjusted Performance Analysis**.

## Features
- Yahoo Finance download or CSV upload
- Four-gene chromosome: Fast EMA period, Slow EMA period, RSI window, trailing stop-loss %
- Genetic algorithm using DEAP: tournament selection, two-point crossover, mutation
- Sharpe-ratio fitness with a light maximum-drawdown penalty
- Chronological train/test split and out-of-sample equity curve
- Convergence chart, buy-and-hold comparison, metrics, CSV downloads

## Run locally
1. Install Python 3.10 or newer.
2. Extract the ZIP file and open a terminal in the project folder.
3. Create/activate a virtual environment (recommended).
4. Install packages:
   `pip install -r requirements.txt`
5. Launch:
   `streamlit run app.py`
6. Streamlit opens the dashboard in your browser.

## Market data
The app downloads data through `yfinance` when you select **Download market data**. This requires internet access and Yahoo Finance data availability. Alternatively, select **Upload CSV** and upload a file with a `Close` column and preferably a `Date` column.

## Important research notes
- The implementation uses a transparent long/flat EMA + RSI rule, with a trailing stop.
- Genetic optimization is run on the training segment only; the later segment is used as out-of-sample test data.
- Annualization currently assumes 252 periods per year, appropriate as a rough daily-bar convention. If you use intraday data, update annualization to reflect the bar interval before interpreting annualized metrics.
- This is a prototype for academic testing, not a live-trading execution system. It does not place orders or connect to a brokerage account.
- Historical performance is not a guarantee of future results. Include spread, slippage, execution, and walk-forward tests before making research conclusions.

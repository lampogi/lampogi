import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from deap import base, creator, tools, algorithms
import random
from datetime import date, timedelta

try:
    import yfinance as yf
except ImportError:
    yf = None

st.set_page_config(page_title="Genetic Trading Strategy Optimizer", page_icon="🧬", layout="wide")

st.title("🧬 Trading Strategy Research Lab")
st.caption("A beginner-friendly tool for testing a rule-based strategy on past price data")
with st.expander("New here? Start with this simple guide", expanded=True):
    st.markdown("""
    **What this app does:** It tries different combinations of indicator settings on historical prices, then checks the selected combination on a separate later period.

    **How to use it:** 1) Choose a market or upload a CSV. 2) Start with the default settings. 3) Click **Run optimization**. 4) Read the **Test period** results first. 5) Compare the strategy with simply holding the asset.

    **Important:** This is a historical simulation for learning and research—not a forecast or a recommendation to trade. A strong backtest can still fail in the future.
    """)
st.warning(
    "Research/backtesting tool only. Results are historical simulations, not financial advice or a guarantee "
    "of future performance. Fees, slippage, liquidity and data quality can materially change results."
)

with st.sidebar:
    st.header("1. Choose price data")
    st.caption("Pick an asset and time interval, or upload your own historical prices.")
    source = st.radio("Data source", ["Download market data", "Upload CSV"], index=0)
    ticker = st.text_input("Ticker symbol", value="SPY", help="Examples: SPY, AAPL, BTC-USD, EURUSD=X")
    interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m"], index=0)
    period = st.selectbox("History", ["1y", "2y", "5y", "10y"], index=1)
    uploaded = st.file_uploader("CSV with Date and Close columns", type=["csv"]) if source == "Upload CSV" else None
    st.header("2. Search settings")
    st.caption("These control how many strategy variations the computer tests. Defaults are fine for a first run.")
    population_size = st.slider("Number of candidates", 10, 100, 30, step=5, help="How many parameter combinations are tested in each round.")
    generations = st.slider("Search rounds", 3, 40, 12, help="More rounds may explore more combinations, but take longer.")
    cx_prob = st.slider("Mixing chance", 0.1, 1.0, 0.7, 0.05, help="Chance that two candidates exchange settings.")
    mut_prob = st.slider("Change chance", 0.05, 0.8, 0.25, 0.05, help="Chance that a candidate gets a small random change.")
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    st.header("3. Simulation settings")
    st.caption("These affect how the historical test is split and how costs are represented.")
    train_pct = st.slider("Earlier data used to find settings (%)", 50, 90, 70, step=5, help="The remaining later data is kept aside to test the chosen settings.")
    fee_bps = st.number_input("Estimated cost per position change (basis points)", 0.0, 100.0, 5.0, 1.0, help="A basis point is 0.01%. This is a simplified cost assumption.")
    risk_free = st.number_input("Reference annual rate (%)", 0.0, 20.0, 0.0, 0.25, help="Used in the risk-adjusted score; leave at 0 if unsure.")
    run = st.button("🚀 Run optimization", type="primary", use_container_width=True)

def load_data():
    if source == "Upload CSV":
        if uploaded is None:
            return None, "Upload a CSV file to continue."
        try:
            raw = pd.read_csv(uploaded)
            lower = {c.lower().strip(): c for c in raw.columns}
            date_col = next((lower[k] for k in ("date", "datetime", "timestamp", "time") if k in lower), None)
            close_col = lower.get("close") or lower.get("adj close") or lower.get("adj_close")
            if close_col is None:
                return None, "CSV needs a Close (or Adj Close) column."
            if date_col:
                raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
                raw = raw.set_index(date_col)
            else:
                raw.index = pd.RangeIndex(len(raw), name="bar")
            df = pd.DataFrame({"Close": pd.to_numeric(raw[close_col], errors="coerce")}, index=raw.index)
            df = df.dropna()
            if len(df) < 100:
                return None, "Please provide at least 100 valid rows for a meaningful test."
            return df, None
        except Exception as exc:
            return None, f"Could not read CSV: {exc}"
    if yf is None:
        return None, "yfinance is not installed. Install requirements.txt or upload a CSV."
    try:
        df = yf.download(ticker.strip(), period=period, interval=interval, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None, "No data returned. Check the ticker, interval, or try uploading a CSV."
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Close" not in df.columns:
            return None, "Downloaded data does not contain a Close column."
        df = df[["Close"]].dropna()
        if len(df) < 100:
            return None, f"Only {len(df)} rows were returned. Choose a longer period or another interval."
        return df, None
    except Exception as exc:
        return None, f"Data download failed: {exc}. Try a different ticker or upload a CSV."

def rsi(series, period):
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def strategy_returns(close, genes, fee_rate=0.0005):
    fast, slow, rsi_period, trail_pct = map(int, genes)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    momentum = rsi(close, rsi_period)
    # Long when trend is bullish and RSI confirms momentum; otherwise flat.
    raw_signal = ((ema_fast > ema_slow) & (momentum > 50)).astype(float)
    # Trailing stop is applied to each long position using the running high since entry.
    prices = close.to_numpy(dtype=float)
    signals = raw_signal.to_numpy(dtype=float)
    position = np.zeros(len(prices), dtype=float)
    in_position = False
    peak = 0.0
    for i in range(1, len(prices)):
        if not in_position and signals[i] > 0:
            in_position, peak = True, prices[i]
        elif in_position:
            peak = max(peak, prices[i])
            if prices[i] <= peak * (1 - trail_pct / 100) or signals[i] <= 0:
                in_position, peak = False, 0.0
        position[i] = 1.0 if in_position else 0.0
    # Avoid look-ahead: position decided on bar i earns the next bar's return.
    asset_returns = close.pct_change().fillna(0).to_numpy(dtype=float)
    held = np.roll(position, 1)
    held[0] = 0
    changes = np.abs(np.diff(np.r_[0.0, held]))
    net_returns = held * asset_returns - changes * fee_rate
    return pd.Series(net_returns, index=close.index, name="Strategy Return"), pd.Series(held, index=close.index, name="Position")

def sharpe(returns, annualization=252, risk_free_annual=0.0):
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 20:
        return -10.0
    excess = r - risk_free_annual / annualization
    vol = excess.std(ddof=1)
    if not np.isfinite(vol) or vol < 1e-10:
        return -10.0
    value = np.sqrt(annualization) * excess.mean() / vol
    return float(value) if np.isfinite(value) else -10.0

def metrics(returns, annualization=252, risk_free_annual=0.0):
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {"Total Return": 0.0, "Annualized Volatility": 0.0, "Sharpe Ratio": 0.0, "Max Drawdown": 0.0, "Trades/Position Changes": 0}
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1
    vol = r.std(ddof=1) * np.sqrt(annualization) if len(r) > 1 else 0
    return {
        "Total Return": float(equity.iloc[-1] - 1),
        "Annualized Volatility": float(vol),
        "Sharpe Ratio": sharpe(r, annualization, risk_free_annual),
        "Max Drawdown": float(drawdown.min()),
        "Trades/Position Changes": int((r != 0).sum())
    }

def run_ga(train_close, pop_size, ngen, cxpb, mutpb, fee, rf, random_seed):
    random.seed(int(random_seed))
    np.random.seed(int(random_seed))
    if not hasattr(creator, "FitnessMaxTrading"):
        creator.create("FitnessMaxTrading", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "TradingIndividual"):
        creator.create("TradingIndividual", list, fitness=creator.FitnessMaxTrading)
    toolbox = base.Toolbox()
    toolbox.register("fast", random.randint, 3, 40)
    toolbox.register("slow", random.randint, 10, 150)
    toolbox.register("rsi_period", random.randint, 5, 30)
    toolbox.register("trail", random.uniform, 1.0, 15.0)
    toolbox.register("individual", tools.initCycle, creator.TradingIndividual,
                     (toolbox.fast, toolbox.slow, toolbox.rsi_period, toolbox.trail), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def repair(ind):
        ind[0] = int(np.clip(round(ind[0]), 3, 40))
        ind[1] = int(np.clip(round(ind[1]), 10, 150))
        ind[2] = int(np.clip(round(ind[2]), 5, 30))
        ind[3] = float(np.clip(ind[3], 1.0, 15.0))
        if ind[0] >= ind[1]:
            ind[1] = min(150, ind[0] + 1)
            if ind[0] >= ind[1]:
                ind[0] = max(3, ind[1] - 1)
        return ind

    def evaluate(ind):
        repaired = repair(ind[:])
        rets, _ = strategy_returns(train_close, repaired, fee)
        score = sharpe(rets, 252, rf)
        # Light penalty for extreme drawdown to discourage fragile high-volatility candidates.
        m = metrics(rets, 252, rf)
        score -= max(0.0, abs(m["Max Drawdown"]) - 0.35) * 0.5
        return (score,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=[3, 10, 5, 1], up=[40, 150, 30, 15], indpb=0.25)

    population = toolbox.population(n=pop_size)
    for ind in population:
        ind[3] = round(float(ind[3]), 2)
        repair(ind)
    hall = tools.HallOfFame(1)
    log = []
    invalid = [i for i in population if not i.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit
    hall.update(population)
    log.append({"Generation": 0, "Best Fitness": max(i.fitness.values[0] for i in population),
                "Mean Fitness": float(np.mean([i.fitness.values[0] for i in population]))})
    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                repair(c1); repair(c2)
                if c1.fitness.valid: del c1.fitness.values
                if c2.fitness.valid: del c2.fitness.values
        for mutant in offspring:
            if random.random() < mutpb:
                # Integer mutation for the first three genes; trail stop is kept numeric.
                before = list(mutant)
                toolbox.mutate(mutant)
                mutant[3] = float(np.clip(before[3] + random.uniform(-3, 3), 1, 15))
                repair(mutant)
                if mutant.fitness.valid: del mutant.fitness.values
        invalid = [i for i in offspring if not i.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit
        population[:] = offspring
        hall.update(population)
        scores = [i.fitness.values[0] for i in population]
        log.append({"Generation": gen, "Best Fitness": max(scores), "Mean Fitness": float(np.mean(scores))})
    best = list(hall[0])
    repair(best)
    return best, pd.DataFrame(log)

def money_pct(v):
    return f"{v:.2%}"

if "result" not in st.session_state:
    st.session_state.result = None

if run:
    data, error = load_data()
    if error:
        st.error(error)
        st.session_state.result = None
    else:
        with st.spinner("Running genetic algorithm and backtesting. This may take a little while..."):
            close = data["Close"].astype(float).sort_index()
            split = int(len(close) * train_pct / 100)
            train = close.iloc[:split]
            test = close.iloc[split:]
            if len(train) < 80 or len(test) < 20:
                st.error("Not enough data after the train/test split. Upload more rows or change the training percentage.")
                st.stop()
            fee = float(fee_bps) / 10000
            best, history = run_ga(train, population_size, generations, cx_prob, mut_prob, fee, float(risk_free)/100, seed)
            train_ret, train_pos = strategy_returns(train, best, fee)
            test_ret, test_pos = strategy_returns(test, best, fee)
            all_ret, all_pos = strategy_returns(close, best, fee)
            st.session_state.result = {
                "close": close, "train": train, "test": test, "best": best, "history": history,
                "train_ret": train_ret, "test_ret": test_ret, "all_ret": all_ret,
                "train_pos": train_pos, "test_pos": test_pos, "all_pos": all_pos,
                "fee": fee, "rf": float(risk_free)/100, "ticker": ticker if source == "Download market data" else "Uploaded CSV",
                "split": split, "source": source
            }

res = st.session_state.result
if res is None:
    st.info("Start with the defaults: choose price data in the left panel, then click **Run optimization**. The app will compare a rule-based strategy with a simple hold approach on later historical data.")
    st.markdown("""
    **What this dashboard does**
    1. Loads historical closing prices from Yahoo Finance or a user-provided CSV.
    2. Splits the data into training and out-of-sample test segments in chronological order.
    3. Evolves four genes: Fast EMA, Slow EMA, RSI window, and trailing stop-loss percentage.
    4. Scores candidate strategies with Sharpe Ratio, with a light drawdown penalty.
    5. Displays convergence, optimized parameters, and train/test equity curves.

    **CSV format:** include a `Date` (or `Datetime`) column and a `Close` column. Date is optional, but recommended.
    """)
else:
    close = res["close"]
    best = res["best"]
    train_ret, test_ret = res["train_ret"], res["test_ret"]
    train_m = metrics(train_ret, 252, res["rf"])
    test_m = metrics(test_ret, 252, res["rf"])
    bm_train = metrics(res["train"].pct_change().fillna(0), 252, res["rf"])
    bm_test = metrics(res["test"].pct_change().fillna(0), 252, res["rf"])

    st.subheader(f"Results for {res['ticker']}")
    st.info("Read the **test period** as the main check: it uses later data that was not used to choose the settings. This still does not guarantee future performance.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fast EMA", int(best[0]))
    c2.metric("Slow EMA", int(best[1]))
    c3.metric("RSI Window", int(best[2]))
    c4.metric("Trailing Stop", f"{float(best[3]):.2f}%")

    st.caption("The genetic algorithm optimizes on the training segment only. The test segment is reserved for out-of-sample evaluation.")
    tab1, tab2, tab3, tab4 = st.tabs(["Results explained", "Search progress", "How the rules work", "Price data"])
    with tab1:
        st.markdown("#### Later-period test: how the selected rules did on data held aside")
        st.caption("The test period is not used to choose the settings. Percentages describe this historical sample only.")
        cols = st.columns(5)
        cols[0].metric("Test total return", money_pct(test_m["Total Return"]))
        cols[1].metric("Risk-adjusted score (Sharpe)", f"{test_m['Sharpe Ratio']:.2f}", help="A higher value means return was larger relative to ups and downs in this sample; it is not a profit guarantee.")
        cols[2].metric("Largest peak-to-trough drop", money_pct(test_m["Max Drawdown"]), help="The biggest decline from a previous high during the test period.")
        cols[3].metric("Price-return variability (annualized)", money_pct(test_m["Annualized Volatility"]), help="A rough measure of how much returns varied; higher means less steady.")
        cols[4].metric("Simple hold return", money_pct(bm_test["Total Return"]), help="A comparison: buy at the start of the test period and hold to the end.")
        equity_strategy = (1 + test_ret).cumprod()
        equity_bh = (1 + res["test"].pct_change().fillna(0)).cumprod()
        eq_df = pd.DataFrame({"Optimized strategy": equity_strategy, "Buy & hold": equity_bh})
        fig = px.line(eq_df, title="Out-of-sample equity curve (starting value = 1.0)", labels={"value": "Equity multiple", "index": "Date"})
        st.plotly_chart(fig, use_container_width=True)
        perf = pd.DataFrame([
            {"Metric": "Total return", "Optimized strategy": money_pct(test_m["Total Return"]), "Buy & hold": money_pct(bm_test["Total Return"])},
            {"Metric": "Sharpe ratio", "Optimized strategy": f"{test_m['Sharpe Ratio']:.2f}", "Buy & hold": f"{bm_test['Sharpe Ratio']:.2f}"},
            {"Metric": "Maximum drawdown", "Optimized strategy": money_pct(test_m["Max Drawdown"]), "Buy & hold": money_pct(bm_test["Max Drawdown"])},
            {"Metric": "Annualized volatility", "Optimized strategy": money_pct(test_m["Annualized Volatility"]), "Buy & hold": money_pct(bm_test["Annualized Volatility"])}
        ])
        st.dataframe(perf, hide_index=True, use_container_width=True)
        st.markdown("#### Earlier-period results (used to choose settings)")
        a, b, c = st.columns(3)
        a.metric("Train return", money_pct(train_m["Total Return"]))
        b.metric("Train Sharpe", f"{train_m['Sharpe Ratio']:.2f}")
        c.metric("Train max drawdown", money_pct(train_m["Max Drawdown"]))
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["history"]["Generation"], y=res["history"]["Best Fitness"], mode="lines+markers", name="Best fitness"))
        fig.add_trace(go.Scatter(x=res["history"]["Generation"], y=res["history"]["Mean Fitness"], mode="lines", name="Mean fitness"))
        fig.update_layout(title="Genetic algorithm convergence", xaxis_title="Generation", yaxis_title="Fitness (risk-adjusted score)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(res["history"].round(4), hide_index=True, use_container_width=True)
    with tab3:
        st.markdown("#### In plain language")
        st.info("The app only models being invested or staying out. It does not simulate short selling.")
        st.write("- **Enter/hold:** The faster average is above the slower average, and the momentum reading is above 50. In this model, that means the rules are positive.")
        st.write("- **Exit/stay out:** The price falls far enough from its highest point since entry (the trailing stop), or the positive entry conditions no longer hold.")
        st.write("- **How candidates are ranked:** by a risk-adjusted score that compares average return with return variability. A small penalty is applied for very large historical declines.")
        st.write("- **Timing and costs:** a signal is applied to the following price interval, and the estimated transaction cost is charged when the position changes.")
        st.write("- **Limitations:** this prototype models long/flat positions only, uses closing-price data, and does not model spread, slippage, funding, dividends beyond adjusted data, or intrabar stop fills.")
        st.markdown("#### Settings selected by the search")
        st.caption("EMA means a moving average that gives more weight to recent prices. RSI is a momentum indicator. The values are numbers of price bars—not automatically days.")
        st.dataframe(pd.DataFrame([
            {"Gene": "Fast EMA period", "Optimized value": int(best[0]), "Search range": "3–40"},
            {"Gene": "Slow EMA period", "Optimized value": int(best[1]), "Search range": "10–150"},
            {"Gene": "RSI window", "Optimized value": int(best[2]), "Search range": "5–30"},
            {"Gene": "Trailing stop-loss", "Optimized value": f"{float(best[3]):.2f}%", "Search range": "1–15%"}
        ]), hide_index=True, use_container_width=True)
        csv = pd.DataFrame({"Close": close, "Strategy_Return": res["all_ret"], "Position": res["all_pos"]}).to_csv().encode("utf-8")
        st.download_button("Download backtest results CSV", data=csv, file_name="trading_ga_backtest.csv", mime="text/csv")
    with tab4:
        st.write(f"Rows: {len(close):,} • Training rows: {len(res['train']):,} • Test rows: {len(res['test']):,}")
        st.dataframe(close.rename("Close").to_frame().tail(100), use_container_width=True)
        st.download_button("Download loaded price data CSV", data=close.rename("Close").to_csv().encode("utf-8"),
                           file_name="loaded_market_data.csv", mime="text/csv")

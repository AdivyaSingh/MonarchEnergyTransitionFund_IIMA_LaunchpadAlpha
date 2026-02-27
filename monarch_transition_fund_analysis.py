
#!/usr/bin/env python3
"""
============================================================
MONARCH INDIA TRANSITION FUND — COMPLETE QUANTITATIVE ANALYSIS
Fund 3 | Thematic: Energy Transition & Industrial Modernisation
Universe: 23 Indian stocks | 3-Year data (2023-2025)
============================================================
OUTPUTS (saved to ./outputs/):
  00. fund_scorecard.csv
  01. universe_cumulative_returns.png
  02. vs_nifty_comparison.png
  03. correlation_heatmap.png
  04. risk_return_scatter.png
  05. rolling_sharpe.png
  06. drawdown_analysis.png
  07. rolling_beta.png
  08. annual_returns_comparison.png
  09. monte_carlo.png
  10. var_distribution.png
  11. sector_returns.png
  12. sharpe_beta.png
  13. portfolio_vs_nifty_annual.png
  14. sortino_calmar.png

Run:  pip install yfinance pandas numpy scipy seaborn matplotlib
      python monarch_transition_fund_analysis.py
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import norm
import yfinance as yf

# ─── OUTPUT DIR ────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

# ─── MATPLOTLIB STYLE: clean professional financial ───────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#F8F9FA",
    "axes.facecolor":    "#FFFFFF",
    "axes.edgecolor":    "#CCCCCC",
    "axes.grid":         True,
    "grid.color":        "#E5E5E5",
    "grid.linewidth":    0.7,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
    "figure.dpi":        150,
    "savefig.dpi":       300,          # all saves at 300 dpi
    "savefig.bbox":      "tight",
    "savefig.facecolor": "#F8F9FA",
})

SAVE_KW = dict(dpi=300, bbox_inches="tight", facecolor="#F8F9FA")

NAVY   = "#2C3E50"
TEAL   = "#16A085"
ORANGE = "#E67E22"
RED    = "#E74C3C"
GREEN  = "#27AE60"
BLUE   = "#3498DB"
PURPLE = "#9B59B6"
GOLD   = "#F39C12"

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — UNIVERSE & DATA
# ═══════════════════════════════════════════════════════════════════════════

TICKERS = {
    # Solar / Wind
    "ADANIGREEN.NS": ("Adani Green",     "Solar/Wind"),
    "SUZLON.NS":     ("Suzlon",          "Solar/Wind"),
    "INOXWIND.NS":   ("Inox Wind",       "Solar/Wind"),
    "WAAREEENER.NS": ("Waaree",          "Solar/Wind"),
    # Power Grid / Transmission
    "POWERGRID.NS":  ("Power Grid",      "Grid/Transmission"),
    "TATAPOWER.NS":  ("Tata Power",      "Grid/Transmission"),
    "ADANIPOWER.NS": ("Adani Power",     "Grid/Transmission"),
    "TORNTPOWER.NS": ("Torrent Power",   "Grid/Transmission"),
    "CESC.NS":       ("CESC",            "Grid/Transmission"),
    # Industrial / EPC
    "ABB.NS":        ("ABB India",       "Industrial/EPC"),
    "SIEMENS.NS":    ("Siemens India",   "Industrial/EPC"),
    "BHEL.NS":       ("BHEL",            "Industrial/EPC"),
    "CUMMINSIND.NS": ("Cummins India",   "Industrial/EPC"),
    "THERMAX.NS":    ("Thermax",         "Industrial/EPC"),
    "KEC.NS":        ("KEC Intl",        "Industrial/EPC"),
    "KALPATPOWR.NS": ("Kalpataru",       "Industrial/EPC"),
    # Energy Efficiency / HVAC
    "HAVELLS.NS":    ("Havells",         "Energy Efficiency"),
    "VOLTAS.NS":     ("Voltas",          "Energy Efficiency"),
    "BLUESTARCO.NS": ("Blue Star",       "Energy Efficiency"),
    # Water / Waste
    "WABAG.NS":      ("VA Tech Wabag",   "Water/Waste"),
    # Diversified Energy
    "NTPC.NS":       ("NTPC",            "Diversified Energy"),
    "WEBSOL.NS":     ("Websol",          "Solar/Wind"),
    # EV / Mobility Components
    "MOTHERSON.NS":  ("Motherson",       "EV/Components"),
}

BENCHMARK    = "^NSEI"
START        = "2020-01-01"
END          = "2025-12-31"
RISK_FREE    = 0.065          # RBI repo rate ~6.5%
TRADING_DAYS = 252

print("=" * 60)
print("MONARCH INDIA TRANSITION FUND — ANALYSIS")
print("=" * 60)
print(f"Universe  : {len(TICKERS)} stocks")
print(f"Period    : {START} → {END}")
print(f"Benchmark : Nifty 50")
print(f"Risk-free : {RISK_FREE*100:.1f}%")
print()

# ─── Download ──────────────────────────────────────────────────────────────
ticker_list  = list(TICKERS.keys()) + [BENCHMARK]
print("Downloading price data via yfinance...")
raw = yf.download(ticker_list, start=START, end=END,
                  auto_adjust=True, progress=True)["Close"]
raw.dropna(how="all", inplace=True)

# Separate benchmark
nifty  = raw[[BENCHMARK]].copy()
stocks = raw.drop(columns=[BENCHMARK])

# Drop any tickers with >20% missing
threshold = 0.20
stocks = stocks.loc[:, stocks.isna().mean() < threshold]
stocks.ffill(inplace=True)
stocks.dropna(how="any", inplace=True)

valid_tickers = list(stocks.columns)
nifty = nifty.loc[stocks.index]
nifty.ffill(inplace=True)

print(f"Valid tickers after quality filter: {len(valid_tickers)}")
print(f"Date range in data: {stocks.index[0].date()} → {stocks.index[-1].date()}")
print(f"Trading days: {len(stocks)}")
print()

# Friendly names & sectors for valid tickers
names   = {t: TICKERS[t][0] for t in valid_tickers if t in TICKERS}
sectors = {t: TICKERS[t][1] for t in valid_tickers if t in TICKERS}

# ─── Returns ───────────────────────────────────────────────────────────────
daily_ret = stocks.pct_change().dropna()
nifty_ret = nifty.pct_change().dropna()

# Align
idx       = daily_ret.index.intersection(nifty_ret.index)
daily_ret = daily_ret.loc[idx]
nifty_ret = nifty_ret.loc[idx].squeeze()

# Equal-weighted portfolio daily return
port_ret   = daily_ret.mean(axis=1)   # EW portfolio

# Cumulative returns
cum_stocks = (1 + daily_ret).cumprod()
cum_nifty  = (1 + nifty_ret).cumprod()
cum_port   = (1 + port_ret).cumprod()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — ANALYTICS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def max_drawdown(ret_series):
    wealth   = (1 + ret_series).cumprod()
    roll_max = wealth.cummax()
    return float(((wealth - roll_max) / roll_max).min())

def drawdown_series(ret_series):
    wealth   = (1 + ret_series).cumprod()
    roll_max = wealth.cummax()
    return (wealth - roll_max) / roll_max

def annual_stats(ret_series, rf=RISK_FREE):
    """Comprehensive annual statistics: Sharpe, Sortino, Calmar, tail risk."""
    T   = TRADING_DAYS
    ann = (1 + ret_series.mean()) ** T - 1
    vol = ret_series.std() * np.sqrt(T)

    sharpe = (ann - rf) / vol if vol > 0 else np.nan

    # Sortino — downside volatility relative to risk-free
    rf_daily     = (1 + rf) ** (1 / T) - 1
    excess_daily = ret_series - rf_daily
    downside     = excess_daily[excess_daily < 0]
    down_vol     = np.sqrt((downside ** 2).mean()) * np.sqrt(T) if len(downside) > 0 else np.nan
    sortino      = (ann - rf) / down_vol if (down_vol and down_vol > 0) else np.nan

    mdd    = max_drawdown(ret_series)
    calmar = ann / abs(mdd) if mdd < 0 else np.nan

    skew  = float(ret_series.skew())
    kurt  = float(ret_series.kurtosis())     # excess kurtosis
    var95 = float(np.percentile(ret_series, 5))
    tail  = ret_series[ret_series <= var95]
    cvar95 = float(tail.mean()) if len(tail) > 0 else var95

    return {
        "ann_ret":   ann,
        "vol":       vol,
        "sharpe":    sharpe,
        "sortino":   sortino,
        "calmar":    calmar,
        "total_ret": float((1 + ret_series).prod() - 1),
        "max_dd":    mdd,
        "skew":      skew,
        "kurt":      kurt,
        "var95":     var95,
        "cvar95":    cvar95,
    }

def beta_calc(stock_ret, bench_ret):
    cov = np.cov(stock_ret.values, bench_ret.values)[0, 1]
    var = np.var(bench_ret.values)
    return cov / var if var > 0 else np.nan

def treynor_ratio(ann_ret, beta_val, rf=RISK_FREE):
    return (ann_ret - rf) / beta_val if (beta_val and beta_val != 0) else np.nan

def information_ratio(port_ret_series, bench_ret_series):
    active  = port_ret_series - bench_ret_series
    te      = active.std() * np.sqrt(TRADING_DAYS)
    ann_act = active.mean() * TRADING_DAYS
    return ann_act / te if te > 0 else np.nan

def year_return(ret_series, year):
    mask = ret_series.index.year == year
    if mask.sum() == 0:
        return np.nan
    return float((1 + ret_series[mask]).prod() - 1)


# ─── Compute stats ─────────────────────────────────────────────────────────
portfolio_stats = annual_stats(port_ret)
nifty_stats     = annual_stats(nifty_ret)

port_beta    = beta_calc(port_ret, nifty_ret)
port_ir      = information_ratio(port_ret, nifty_ret)
port_treynor = treynor_ratio(portfolio_stats["ann_ret"], port_beta)

print("PORTFOLIO (EW) vs. NIFTY 50")
print(f"  Total Return  : Portfolio {portfolio_stats['total_ret']*100:.1f}%"
      f"  |  Nifty {nifty_stats['total_ret']*100:.1f}%")
print(f"  Ann. Return   : Portfolio {portfolio_stats['ann_ret']*100:.1f}%"
      f"  |  Nifty {nifty_stats['ann_ret']*100:.1f}%")
print(f"  Volatility    : Portfolio {portfolio_stats['vol']*100:.1f}%"
      f"  |  Nifty {nifty_stats['vol']*100:.1f}%")
print(f"  Sharpe Ratio  : Portfolio {portfolio_stats['sharpe']:.2f}"
      f"  |  Nifty {nifty_stats['sharpe']:.2f}")
print(f"  Sortino Ratio : Portfolio {portfolio_stats['sortino']:.2f}"
      f"  |  Nifty {nifty_stats['sortino']:.2f}")
print(f"  Calmar Ratio  : Portfolio {portfolio_stats['calmar']:.2f}"
      f"  |  Nifty {nifty_stats['calmar']:.2f}")
print(f"  Max Drawdown  : Portfolio {portfolio_stats['max_dd']*100:.1f}%"
      f"  |  Nifty {nifty_stats['max_dd']*100:.1f}%")
print(f"  Beta          : {port_beta:.2f}")
print(f"  Info. Ratio   : {port_ir:.2f}")
print(f"  Treynor Ratio : {port_treynor:.4f}")
print()

# Individual stock stats
stock_stats = {t: annual_stats(daily_ret[t]) for t in valid_tickers}
betas       = {t: beta_calc(daily_ret[t], nifty_ret) for t in valid_tickers}

years        = [2023, 2024, 2025]
port_annual  = {y: year_return(port_ret,  y) for y in years}
nifty_annual = {y: year_return(nifty_ret, y) for y in years}

# Sector colours (shared across charts)
sector_colors = {
    "Solar/Wind":        "#F39C12",
    "Grid/Transmission": "#3498DB",
    "Industrial/EPC":    "#2ECC71",
    "Energy Efficiency": "#9B59B6",
    "Water/Waste":       "#1ABC9C",
    "Diversified Energy":"#E74C3C",
    "EV/Components":     "#E67E22",
}


# ═══════════════════════════════════════════════════════════════════════════
# CHART 1 — CUMULATIVE RETURNS OF ALL STOCKS
# ═══════════════════════════════════════════════════════════════════════════
print("Chart 1: Universe Cumulative Returns...")
fig, ax = plt.subplots(figsize=(14, 7))

# Use non-deprecated colormap API
cmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(valid_tickers))
for i, t in enumerate(valid_tickers):
    ax.plot(cum_stocks.index, cum_stocks[t].values,
            alpha=0.45, lw=0.9, color=cmap(i), label=names.get(t, t))

ax.plot(cum_port.index, cum_port.values, color=NAVY, lw=2.8,
        label="EW Portfolio", zorder=5)
ax.plot(cum_nifty.index, cum_nifty.values, color=RED, lw=2.2,
        linestyle="--", label="Nifty 50", zorder=5)

ax.axhline(1.0, color="#999999", lw=0.8, linestyle=":")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.1f}x"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.set_title("Monarch Transition Fund — Cumulative Returns (2023–2025)")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value (Base = 1.0×)")
ax.legend(loc="upper left", ncol=3, fontsize=7)
plt.tight_layout()
plt.savefig("outputs/01_universe_cumulative_returns.png", **SAVE_KW)
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# CHART 2 — EW PORTFOLIO vs NIFTY (Area + Relative)
# ═══════════════════════════════════════════════════════════════════════════
print("Chart 2: Portfolio vs Nifty...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                gridspec_kw={"height_ratios": [3, 1]},
                                sharex=True)

ax1.fill_between(cum_port.index, 1, cum_port.values, alpha=0.18, color=NAVY)
ax1.plot(cum_port.index, cum_port.values, color=NAVY, lw=2.5,
         label="Transition Portfolio (EW)")
ax1.fill_between(cum_nifty.index, 1, cum_nifty.values, alpha=0.12, color=RED)
ax1.plot(cum_nifty.index, cum_nifty.values, color=RED, lw=2.0,
         linestyle="--", label="Nifty 50")
ax1.axhline(1.0, color="#999999", lw=0.7, linestyle=":")
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.1f}x"))
ax1.set_title("Transition Fund vs Nifty 50 — Cumulative Performance (2023–2025)")
ax1.set_ylabel("Portfolio Value (Base = 1.0×)")
ax1.legend(loc="upper left")

# Relative performance (excess return ratio)
excess = cum_port.values / cum_nifty.values.squeeze()
ax2.fill_between(cum_port.index, 1, excess,
                 where=(excess >= 1), alpha=0.4, color=GREEN,
                 label="Outperformance")
ax2.fill_between(cum_port.index, 1, excess,
                 where=(excess < 1),  alpha=0.4, color=RED,
                 label="Underperformance")
ax2.axhline(1.0, color="#999999", lw=0.8)
ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.2f}×"))
ax2.set_ylabel("Relative to Nifty")
ax2.set_xlabel("Date")
ax2.legend(loc="upper left", fontsize=8)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig("outputs/02_vs_nifty_comparison.png", **SAVE_KW)
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# CHART 3 — CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════
print("Chart 3: Correlation Heatmap...")
corr_df = daily_ret.copy()
corr_df.columns = [names.get(t, t) for t in corr_df.columns]
corr = corr_df.corr()

fig, ax = plt.subplots(figsize=(14, 12))
# Lower-triangle only (keep diagonal for reference)
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
cmap_heat = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap=cmap_heat, center=0, vmin=-0.3, vmax=1.0,
            linewidths=0.4, ax=ax, annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8, "label": "Pearson r"})
ax.set_title("Return Correlation Matrix — Transition Fund Universe (2023–2025)")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("outputs/03_correlation_heatmap.png", **SAVE_KW)
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# CHART 4 — RISK-RETURN SCATTER (with correct CML)
# ═══════════════════════════════════════════════════════════════════════════
print("Chart 4: Risk-Return Scatter...")
fig, ax = plt.subplots(figsize=(12, 7))

sector_colors = {
    "Solar/Wind":        "#F39C12",
    "Grid/Transmission": "#3498DB",
    "Industrial/EPC":    "#2ECC71",
    "Energy Efficiency": "#9B59B6",
    "Water/Waste":       "#1ABC9C",
    "Diversified Energy":"#E74C3C",
    "EV/Components":     "#E67E22",
}

for t in valid_tickers:
    s   = stock_stats[t]
    nm  = names.get(t, t)
    sec = sectors.get(t, "Other")
    col = sector_colors.get(sec, NAVY)
    ax.scatter(s["vol"]*100, s["ann_ret"]*100, s=140, color=col, alpha=0.85,
               edgecolors="white", lw=0.8, zorder=3)
    ax.annotate(nm, (s["vol"]*100, s["ann_ret"]*100),
                xytext=(4, 3), textcoords="offset points", fontsize=7)

# Portfolio & Nifty
ax.scatter(portfolio_stats["vol"]*100, portfolio_stats["ann_ret"]*100,
           s=300, color=NAVY, marker="*", zorder=5, label="EW Portfolio")
ax.scatter(nifty_stats["vol"]*100, nifty_stats["ann_ret"]*100,
           s=220, color=RED, marker="D", zorder=5, label="Nifty 50")

# Capital Market Line: y = rf + Sharpe * vol (passes through rf at vol=0)
rf_pct  = RISK_FREE * 100
max_vol = max(s["vol"]*100 for s in stock_stats.values()) + 5
cml_x   = np.linspace(0, max_vol, 200)
cml_y   = rf_pct + portfolio_stats["sharpe"] * cml_x
ax.plot(cml_x, cml_y, color="#AAAAAA", lw=1.3, linestyle="--",
        label=f"CML (Sharpe={portfolio_stats['sharpe']:.2f})")
ax.axhline(rf_pct, color="#AAAAAA", lw=0.8, linestyle=":",
           label=f"Risk-free ({RISK_FREE*100:.1f}%)")
ax.axhline(0, color="#CCCCCC", lw=0.6)

patches = [mpatches.Patch(color=v, label=k) for k, v in sector_colors.items()]
patches += [mpatches.Patch(color=NAVY, label="EW Portfolio"),
            mpatches.Patch(color=RED,  label="Nifty 50")]
ax.legend(handles=patches, loc="lower right", fontsize=8, ncol=2)
ax.set_title("Risk-Return Scatter — Transition Fund Universe (3-Year Ann.)")
ax.set_xlabel("Annualised Volatility (%)")
ax.set_ylabel("Annualised Return (%)")
plt.tight_layout()
plt.savefig("outputs/04_risk_return_scatter.png", **SAVE_KW)
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# CHART 5 — ROLLING 90-DAY SHARPE RATIO
# ═══════════════════════════════════════════════════════════════════════════
print("Chart 5: Rolling Sharpe...")
WINDOW = 90
roll_ann_p    = port_ret.rolling(WINDOW).mean() * TRADING_DAYS
roll_vol_p    = port_ret.rolling(WINDOW).std()  * np.sqrt(TRADING_DAYS)
roll_sharpe_p = (roll_ann_p - RISK_FREE) / roll_vol_p

roll_ann_n    = nifty_ret.rolling(WINDOW).mean() * TRADING_DAYS
roll_vol_n    = nifty_ret.rolling(WINDOW).std()  * np.sqrt(TRADING_DAYS)
roll_sharpe_n = (roll_ann_n - RISK_FREE) / roll_vol_n

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(roll_sharpe_p.index, roll_sharpe_p.values,
        color=NAVY, lw=2.0, label="Transition Portfolio")
ax.plot(roll_sharpe_n.index, roll_sharpe_n.values,
        color=RED, lw=1.6, linestyle="--", label="Nifty 50")
ax.axhline(0, color="#999999", lw=0.8)
ax.axhline(1, color=GREEN, lw=0.8, linestyle=":", alpha=0.7, label="Sharpe = 1")

# Use boolean masks to prevent NaN gaps in fill_between
pos_mask = roll_sharpe_p.fillna(0) >= 0
neg_mask = roll_sharpe_p.fillna(0) < 0
ax.fill_between(roll_sharpe_p.index, 0, roll_sharpe_p.values,
                where=pos_mask.values, alpha=0.15, color=GREEN)
ax.fill_between(roll_sharpe_p.index, roll_sharpe_p.values, 0,
                where=neg_mask.values, alpha=0.15, color=RED)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.set_title(f"Rolling {WINDOW}-Day Sharpe Ratio — Portfolio vs Nifty")
ax.set_ylabel("Sharpe Ratio (Annualised)")
ax.set_xlabel("Date")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/05_rolling_sharpe.png", **SAVE_KW)
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# CHART 6 — DRAWDOWN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("Chart 6: Drawdown Analysis...")
def drawdown_series(ret_series):
    wealth    = (1 + ret_series).cumprod()
    roll_max  = wealth.cummax()
    return (wealth - roll_max) / roll_max

dd_port  = drawdown_series(port_ret)
dd_nifty = drawdown_series(nifty_ret)

fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(dd_port.index, dd_port.values * 100, 0,
                alpha=0.50, color=NAVY, label="Portfolio Drawdown")
ax.fill_between(dd_nifty.index, dd_nifty.values * 100, 0,
                alpha=0.28, color=RED, label="Nifty Drawdown")
ax.plot(dd_port.index,  dd_port.values  * 100, color=NAVY, lw=1.5)
ax.plot(dd_nifty.index, dd_nifty.values * 100, color=RED,  lw=1.2, linestyle="--")

# Annotate worst drawdown
mdd_idx = dd_port.idxmin()
ax.annotate(f"  Max DD\n  {dd_port.min()*100:.1f}%",
            xy=(mdd_idx, dd_port.min() * 100),
            xytext=(mdd_idx, dd_port.min() * 100 - 3),
            fontsize=8, color=NAVY,
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.8))

ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.set_title("Drawdown from Peak — Portfolio vs Nifty (2023–2025)")
ax.set_ylabel("Drawdown (%)")
ax.set_xlabel("Date")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/06_drawdown_analysis.png", **SAVE_KW)
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# CHART 7 — ROLLING BETA (90-day)
# ═══════════════════════════════════════════════════════════════════════════
print("Chart 7: Rolling Beta...")
BETA_WIN  = 90
roll_cov  = port_ret.rolling(BETA_WIN).cov(nifty_ret)
roll_var  = nifty_ret.rolling(BETA_WIN).var()
roll_beta = roll_cov / roll_var

mean_beta = float(roll_beta.dropna().mean())

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(roll_beta.index, roll_beta.values, color=TEAL, lw=2.0,
        label=f"Portfolio Beta (90d rolling, mean={mean_beta:.2f})")
ax.axhline(1.0, color=RED,       lw=1.0, linestyle="--", label="β = 1.0 (market)")
ax.axhline(mean_beta, color=TEAL, lw=0.9, linestyle=":", alpha=0.7,
           label=f"Mean β = {mean_beta:.2f}")
ax.axhline(0.0, color="#999999", lw=0.6)

high_mask = roll_beta.fillna(1) >= 1
low_mask  = roll_beta.fillna(1) < 1
ax.fill_between(roll_beta.index, 1.0, roll_beta.values,
                where=high_mask.values, alpha=0.18, color=ORANGE,
                label="β > 1 (high market sensitivity)")
ax.fill_between(roll_beta.index, 1.0, roll_beta.values,
                where=low_mask.values, alpha=0.15, color=TEAL,
                label="β < 1 (defensive)")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.set_title("Rolling 90-Day Beta vs Nifty 50 — Transition Portfolio")
ax.set_ylabel("Beta")
ax.set_xlabel("Date")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/07_rolling_beta.png", **SAVE_KW)
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# CHART 8 — ANNUAL RETURNS COMPARISON (Horizontal Bar)
# ═══════════════════════════════════════════════════════════════════════════
print("Chart 8: Annual Returns Comparison...")
all_annual = {}
for t in valid_tickers:
    all_annual[names.get(t, t)] = [year_return(daily_ret[t], y)*100 for y in years]
all_annual["EW Portfolio"] = [port_annual[y]*100 for y in years]
all_annual["Nifty 50"]     = [nifty_annual[y]*100 for y in years]

annual_df = pd.DataFrame(all_annual, index=[str(y) for y in years]).T
annual_df = annual_df.sort_values("2023", ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=False)
for i, yr in enumerate(["2023", "2024", "2025"]):
    ser = annual_df[yr].dropna().sort_values(ascending=True)
    colors = [NAVY if n == "EW Portfolio"
              else RED  if n == "Nifty 50"
              else GREEN if v >= 0
              else "#E74C3C"
              for n, v in ser.items()]
    axes[i].barh(ser.index, ser.values, color=colors, edgecolor="white", lw=0.4)
    axes[i].axvline(0, color="#999999", lw=0.7)
    nifty_val = nifty_annual[int(yr)] * 100
    axes[i].axvline(nifty_val, color=RED, lw=1.2, linestyle="--",
                    alpha=0.7, label=f"Nifty {nifty_val:.1f}%")
    axes[i].set_title(f"Returns {yr}")
    axes[i].set_xlabel("Return (%)")
    axes[i].legend(fontsize=8)
    if i > 0:
        axes[i].set_yticks([])
plt.suptitle("Annual Returns by Stock vs Nifty — Transition Fund Universe",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/08_annual_returns_comparison.png", **SAVE_KW)
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# CHART 9 — MONTE CARLO SIMULATION (5-year, Geometric Brownian Motion)
# ═══════════════════════════════════════════════════════════════════════════
print("Chart 9: Monte Carlo Simulation (GBM)...")
n_sims    = 5000
n_days    = TRADING_DAYS * 5   # 5-year horizon
mu_daily  = port_ret.mean()
sig_daily = port_ret.std()

# GBM log-returns drift: μ_log = μ - σ²/2  (removes log-normal upward bias)
mu_log = mu_daily - 0.5 * sig_daily ** 2

np.random.seed(42)
# Fully vectorised: shape (n_days, n_sims)
log_shocks  = np.random.normal(mu_log, sig_daily, size=(n_days, n_sims))
log_paths   = np.vstack([np.zeros(n_sims), np.cumsum(log_shocks, axis=0)])
sims        = np.exp(log_paths)   # price paths starting at 1.0

final_vals = sims[-1, :]
p5, p25, p50, p75, p95 = np.percentile(final_vals, [5, 25, 50, 75, 95])

# Nifty GBM median path (deterministic)
nifty_mu_log   = nifty_ret.mean() - 0.5 * nifty_ret.std() ** 2
nifty_gbm_path = np.exp(nifty_mu_log * np.arange(n_days + 1))

prob_loss     = (final_vals < 1.0).mean() * 100
prob_vs_nifty = (final_vals > nifty_gbm_path[-1]).mean() * 100

fig, ax = plt.subplots(figsize=(13, 7))
x = np.arange(n_days + 1)
ax.fill_between(x, np.percentile(sims, 5,  axis=1),
                   np.percentile(sims, 95, axis=1),
                   alpha=0.10, color=NAVY, label="5th–95th Percentile")
ax.fill_between(x, np.percentile(sims, 25, axis=1),
                   np.percentile(sims, 75, axis=1),
                   alpha=0.20, color=NAVY, label="25th–75th Percentile")
ax.plot(x, np.percentile(sims, 50, axis=1),
        color=NAVY, lw=2.5, label=f"Median P50 = {p50:.2f}x")
ax.plot(x, np.percentile(sims, 5, axis=1),
        color=RED, lw=1.2, linestyle=":", label=f"Bear P5 = {p5:.2f}x")
ax.plot(x, np.percentile(sims, 95, axis=1),
        color=GREEN, lw=1.2, linestyle=":", label=f"Bull P95 = {p95:.2f}x")
ax.plot(x, nifty_gbm_path, color=RED, lw=1.8, linestyle="--",
        label=f"Nifty GBM median = {nifty_gbm_path[-1]:.2f}x")
ax.axhline(1.0, color="#999999", lw=0.7, label="Break-even (1.0x)")
ax.set_xticks(np.arange(0, n_days + 1, TRADING_DAYS))
ax.set_xticklabels([f"Y{i}" for i in range(6)])
ax.set_title(
    f"Monte Carlo GBM — Transition Portfolio 5-Year Projection  "
    f"[N={n_sims:,} | Ann.μ={mu_daily*TRADING_DAYS*100:.1f}% | "
    f"Ann.σ={sig_daily*TRADING_DAYS**0.5*100:.1f}%]\n"
    f"P(Beat Nifty)={prob_vs_nifty:.0f}%  |  P(Loss after 5Y)={prob_loss:.0f}%"
)
ax.set_ylabel("Portfolio Value (Base = 1.0×)")
ax.set_xlabel("Year")
ax.legend(loc="upper left", ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig("outputs/09_monte_carlo.png", **SAVE_KW)
plt.close()

print(f"  Monte Carlo GBM Results (5-year):")
print(f"    Bear  P5  : {p5:.2f}x  ({(p5-1)*100:+.0f}%)")
print(f"    Base  P50 : {p50:.2f}x  ({(p50-1)*100:+.0f}%)")
print(f"    Bull  P95 : {p95:.2f}x  ({(p95-1)*100:+.0f}%)")
print(f"    P(beat Nifty GBM) : {prob_vs_nifty:.0f}%")
print(f"    P(loss after 5Y)  : {prob_loss:.0f}%")
print()


# ═══════════════════════════════════════════════════════════════════════════
# CHART 10 — VaR & CVaR DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
print("Chart 10: VaR Distribution...")
fig, ax = plt.subplots(figsize=(11, 6))

ret_arr = port_ret.dropna().values
var95   = float(np.percentile(ret_arr, 5))
cvar95  = float(ret_arr[ret_arr <= var95].mean())

n_counts, bins, patches_hist = ax.hist(
    ret_arr * 100, bins=80, color=NAVY, edgecolor="white",
    linewidth=0.3, alpha=0.72, density=True, label="Daily Returns"
)
# Shade the tail red
for patch, left in zip(patches_hist, bins[:-1]):
    if left / 100 <= var95:
        patch.set_facecolor(RED)
        patch.set_alpha(0.55)

# Normal fit overlay
mu_fit, sigma_fit = norm.fit(ret_arr)
x_line = np.linspace(ret_arr.min() * 100, ret_arr.max() * 100, 400)
ax.plot(x_line, norm.pdf(x_line / 100, mu_fit, sigma_fit) / 100,
        color=ORANGE, lw=2.0, label="Normal Fit")

ax.axvline(var95  * 100, color=RED,       lw=2.0, linestyle="--",
           label=f"VaR 95% = {var95*100:.2f}%")
ax.axvline(cvar95 * 100, color="#8B0000", lw=2.0, linestyle=":",
           label=f"CVaR 95% = {cvar95*100:.2f}%")

ax.set_title("Daily Return Distribution — Transition Portfolio (VaR & CVaR)")
ax.set_xlabel("Daily Return (%)")
ax.set_ylabel("Density")
ax.legend(fontsize=9)

info_text = (
    f"Skewness  : {port_ret.skew():.3f}\n"
    f"Ex.Kurt   : {port_ret.kurtosis():.3f}\n"
    f"Hit Rate  : {(ret_arr > 0).mean()*100:.0f}%\n"
    f"Ann. VaR  : {var95 * np.sqrt(TRADING_DAYS) * 100:.2f}%"
)
ax.text(0.98, 0.97, info_text, transform=ax.transAxes, fontsize=9,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
plt.tight_layout()
plt.savefig("outputs/10_var_distribution.png", **SAVE_KW)
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# CHART 11 — SECTOR AVERAGE RETURN
# ═══════════════════════════════════════════════════════════════════════════
print("Chart 11: Sector Contribution...")
sector_returns = {}
for t in valid_tickers:
    sec = sectors.get(t, "Other")
    r   = stock_stats[t]["ann_ret"] * 100
    if sec not in sector_returns:
        sector_returns[sec] = []
    sector_returns[sec].append(r)

sector_avg = {s: np.mean(v) for s, v in sector_returns.items()}
sector_avg = dict(sorted(sector_avg.items(), key=lambda x: x[1], reverse=True))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(list(sector_avg.keys()), list(sector_avg.values()),
              color=[sector_colors.get(s, NAVY) for s in sector_avg.keys()],
              edgecolor="white", lw=0.5)
ax.axhline(nifty_stats["ann_ret"]*100, color=RED, lw=1.5, linestyle="--",
           label=f"Nifty 50 ({nifty_stats['ann_ret']*100:.1f}%)")
ax.axhline(0, color="#999999", lw=0.7)
for bar, val in zip(bars, sector_avg.values()):
    ypos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 2.0
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_title("Average Annual Return by Sector vs Nifty (3-Year Ann.)")
ax.set_ylabel("Avg Annual Return (%)")
ax.set_xlabel("Sector")
ax.legend()
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("outputs/11_sector_returns.png", **SAVE_KW)
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# CHART 12 — INDIVIDUAL STOCK SHARPE & BETA (Bubble Chart)
# ═══════════════════════════════════════════════════════════════════════════
print("Chart 12: Sharpe & Beta per Stock...")
fig, ax = plt.subplots(figsize=(12, 7))

for t in valid_tickers:
    s    = stock_stats[t]
    nm   = names.get(t, t)
    sec  = sectors.get(t, "Other")
    col  = sector_colors.get(sec, NAVY)
    b    = betas.get(t, 1.0)
    size = max(abs(s["ann_ret"]) * 500, 60)  # bubble size = |return|
    ax.scatter(b, s["sharpe"], s=size, color=col, alpha=0.80,
               edgecolors="white", lw=0.8)
    ax.annotate(nm, (b, s["sharpe"]),
                xytext=(4, 3), textcoords="offset points", fontsize=7)

# True portfolio beta on x-axis
ax.scatter(port_beta, portfolio_stats["sharpe"],
           s=300, color=NAVY, marker="*", zorder=5, label="EW Portfolio")
ax.scatter(1.0, nifty_stats["sharpe"],
           s=200, color=RED, marker="D", zorder=5, label="Nifty 50")

ax.axhline(0, color="#999999", lw=0.7)
ax.axhline(1, color=GREEN, lw=0.8, linestyle=":", alpha=0.7, label="Sharpe = 1")
ax.axvline(1, color=RED,   lw=0.8, linestyle=":", alpha=0.7, label="Beta = 1")

patches = [mpatches.Patch(color=v, label=k) for k, v in sector_colors.items()]
patches += [
    mpatches.Patch(color=NAVY,  label="EW Portfolio"),
    mpatches.Patch(color=RED,   label="Nifty 50 / β=1"),
    mpatches.Patch(color=GREEN, label="Sharpe = 1"),
]
ax.legend(handles=patches, fontsize=8, ncol=2)
ax.set_title("Sharpe Ratio vs Beta — Transition Fund Universe (3-Year Ann.)")
ax.set_xlabel("Beta vs Nifty 50")
ax.set_ylabel("Sharpe Ratio (Annualised)")
plt.tight_layout()
plt.savefig("outputs/12_sharpe_beta.png", **SAVE_KW)
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# CHART 13 — PORTFOLIO ANNUAL RETURN vs NIFTY (Bar comparison)
# ═══════════════════════════════════════════════════════════════════════════
print("Chart 13: Annual Portfolio vs Nifty...")
fig, ax = plt.subplots(figsize=(8, 5))
x      = np.arange(len(years))
width  = 0.35
vals_p = [port_annual[y]*100 for y in years]
vals_n = [nifty_annual[y]*100 for y in years]

bars1 = ax.bar(x - width/2, vals_p, width, color=NAVY, label="Transition Portfolio (EW)")
bars2 = ax.bar(x + width/2, vals_n, width, color=RED, alpha=0.80, label="Nifty 50")

all_bars = list(bars1) + list(bars2)
all_vals = vals_p + vals_n
for bar, val in zip(all_bars, all_vals):
    if val >= 0:
        ypos, va = bar.get_height() + 0.5, "bottom"
    else:
        ypos, va = bar.get_height() - 1.5, "top"
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f"{val:.1f}%", ha="center", va=va, fontsize=9, fontweight="bold")

ax.axhline(0, color="#999999", lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in years])
ax.set_ylabel("Annual Return (%)")
ax.set_title("Calendar Year Returns — Transition Portfolio vs Nifty")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/13_portfolio_vs_nifty_annual.png", **SAVE_KW)
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# CHART 14 — SORTINO, CALMAR & SHARPE PER STOCK
# ═══════════════════════════════════════════════════════════════════════════
print("Chart 14: Sortino / Calmar / Sharpe per Stock...")
stock_names_sorted = sorted(valid_tickers,
                             key=lambda t: stock_stats[t]["sortino"],
                             reverse=True)
nm_labels = [names.get(t, t) for t in stock_names_sorted]
sortino_v = [stock_stats[t]["sortino"] for t in stock_names_sorted]
calmar_v  = [stock_stats[t]["calmar"]  for t in stock_names_sorted]
sharpe_v  = [stock_stats[t]["sharpe"]  for t in stock_names_sorted]

fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
metrics = [
    (sortino_v, "Sortino Ratio",  TEAL,   "sortino"),
    (calmar_v,  "Calmar Ratio",   PURPLE, "calmar"),
    (sharpe_v,  "Sharpe Ratio",   ORANGE, "sharpe"),
]
for ax, (vals, title, color, key) in zip(axes, metrics):
    bar_colors = [color if v >= 0 else RED for v in vals]
    ax.barh(nm_labels, vals, color=bar_colors, edgecolor="white", lw=0.4)
    ax.axvline(0, color="#999999", lw=0.7)
    port_val = portfolio_stats[key]
    ax.axvline(port_val, color=NAVY, lw=1.5, linestyle="--",
               label=f"Portfolio: {port_val:.2f}")
    ax.set_title(title)
    ax.set_xlabel(title)
    ax.legend(fontsize=8)

plt.suptitle("Risk-Adjusted Ratios — Transition Fund Universe (3-Year Ann.)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/14_sortino_calmar.png", **SAVE_KW)
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# SCORECARD CSV (extended with Sortino, Calmar, Treynor, IR)
# ═══════════════════════════════════════════════════════════════════════════
print("Scorecard CSV...")
rows = []
for t in valid_tickers:
    s   = stock_stats[t]
    b   = betas.get(t, np.nan)
    nm  = names.get(t, t)
    sec = sectors.get(t, "Other")
    rows.append({
        "Ticker":          t,
        "Name":            nm,
        "Sector":          sec,
        "Ann Return (%)":  round(s["ann_ret"]*100, 1),
        "Total Ret (%)":   round(s["total_ret"]*100, 1),
        "Volatility (%)":  round(s["vol"]*100, 1),
        "Sharpe":          round(s["sharpe"], 2),
        "Sortino":         round(s["sortino"], 2),
        "Calmar":          round(s["calmar"], 2),
        "Treynor":         round(treynor_ratio(s["ann_ret"], b), 4),
        "Beta":            round(b, 2),
        "Max DD (%)":      round(s["max_dd"]*100, 1),
        "Skewness":        round(s["skew"], 2),
        "Ex.Kurtosis":     round(s["kurt"], 2),
        "VaR 95% (daily)": round(s["var95"]*100, 3),
        "CVaR 95% (daily)":round(s["cvar95"]*100, 3),
        "Y2023 (%)":       round(year_return(daily_ret[t], 2023)*100, 1),
        "Y2024 (%)":       round(year_return(daily_ret[t], 2024)*100, 1),
        "Y2025 (%)":       round(year_return(daily_ret[t], 2025)*100, 1),
    })

scorecard = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)

# Add portfolio row
prow = {
    "Ticker":          "PORTFOLIO",
    "Name":            "EW Portfolio",
    "Sector":          "All",
    "Ann Return (%)":  round(portfolio_stats["ann_ret"]*100, 1),
    "Total Ret (%)":   round(portfolio_stats["total_ret"]*100, 1),
    "Volatility (%)":  round(portfolio_stats["vol"]*100, 1),
    "Sharpe":          round(portfolio_stats["sharpe"], 2),
    "Sortino":         round(portfolio_stats["sortino"], 2),
    "Calmar":          round(portfolio_stats["calmar"], 2),
    "Treynor":         round(port_treynor, 4),
    "Beta":            round(port_beta, 2),
    "Max DD (%)":      round(portfolio_stats["max_dd"]*100, 1),
    "Skewness":        round(port_ret.skew(), 2),
    "Ex.Kurtosis":     round(port_ret.kurtosis(), 2),
    "VaR 95% (daily)": round(portfolio_stats["var95"]*100, 3),
    "CVaR 95% (daily)":round(portfolio_stats["cvar95"]*100, 3),
    "Y2023 (%)":       round(port_annual[2023]*100, 1),
    "Y2024 (%)":       round(port_annual[2024]*100, 1),
    "Y2025 (%)":       round(port_annual[2025]*100, 1),
}
nrow = {
    "Ticker":          "NIFTY50",
    "Name":            "Nifty 50",
    "Sector":          "Benchmark",
    "Ann Return (%)":  round(nifty_stats["ann_ret"]*100, 1),
    "Total Ret (%)":   round(nifty_stats["total_ret"]*100, 1),
    "Volatility (%)":  round(nifty_stats["vol"]*100, 1),
    "Sharpe":          round(nifty_stats["sharpe"], 2),
    "Sortino":         round(nifty_stats["sortino"], 2),
    "Calmar":          round(nifty_stats["calmar"], 2),
    "Treynor":         round(treynor_ratio(nifty_stats["ann_ret"], 1.0), 4),
    "Beta":            1.00,
    "Max DD (%)":      round(nifty_stats["max_dd"]*100, 1),
    "Skewness":        round(nifty_ret.skew(), 2),
    "Ex.Kurtosis":     round(nifty_ret.kurtosis(), 2),
    "VaR 95% (daily)": round(nifty_stats["var95"]*100, 3),
    "CVaR 95% (daily)":round(nifty_stats["cvar95"]*100, 3),
    "Y2023 (%)":       round(nifty_annual[2023]*100, 1),
    "Y2024 (%)":       round(nifty_annual[2024]*100, 1),
    "Y2025 (%)":       round(nifty_annual[2025]*100, 1),
}

scorecard = pd.concat([pd.DataFrame([prow, nrow]), scorecard], ignore_index=True)
scorecard.to_csv("outputs/00_fund_scorecard.csv", index=False)

print()
print("=" * 70)
print("FINAL SUMMARY — TRANSITION PORTFOLIO vs NIFTY")
print("=" * 70)
print(scorecard[["Name", "Ann Return (%)", "Volatility (%)", "Sharpe",
                  "Sortino", "Calmar", "Max DD (%)", "Beta",
                  "Y2023 (%)", "Y2024 (%)", "Y2025 (%)"]].head(12).to_string(index=False))
print()
print("All 14 outputs saved → ./outputs/")
print("=" * 70)

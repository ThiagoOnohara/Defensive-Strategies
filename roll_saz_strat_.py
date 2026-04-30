# -*- coding: utf-8 -*-

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from datetime import datetime

sys.path.append('F:/Front/Moedas/Base/')
from FX import database

warnings.filterwarnings(action='ignore', category=FutureWarning)

# ============================================================
# DATA LOADING
# ============================================================

def get_ccy_prices():
    currencies = {
        'USDBRLCR CMPN Curncy': 'USDBRL', 'USDMXNCR CMPN Curncy': 'USDMXN',
        'USDCLPCR CMPN Curncy': 'USDCLP', 'USDCOPCR CMPN Curncy': 'USDCOP',
        'USDCNYCR CMPN Curncy': 'USDCNY', 'USDINRCR CMPN Curncy': 'USDINR',
        'USDKRWCR CMPN Curncy': 'USDKRW', 'USDPHPCR CMPN Curncy': 'USDPHP',
        'USDSGDCR CMPN Curncy': 'USDSGD', 'USDIDRCR CMPN Curncy': 'USDIDR',
        'USDTHBCR CMPN Curncy': 'USDTHB', 'USDTWDCR CMPN Curncy': 'USDTWD',
        'USDCZKCR CMPN Curncy': 'USDCZK', 'USDHUFCR CMPN Curncy': 'USDHUF',
        'USDPLNCR CMPN Curncy': 'USDPLN', 'USDZARCR CMPN Curncy': 'USDZAR',
        'AUDUSDCR CMPN Curncy': 'USDAUD', 'USDCADCR CMPN Curncy': 'USDCAD',
        'EURUSDCR CMPN Curncy': 'USDEUR', 'GBPUSDCR CMPN Curncy': 'USDGBP',
        'USDCHFCR CMPN Curncy': 'USDCHF', 'USDJPYCR CMPN Curncy': 'USDJPY',
        'USDNOKCR CMPN Curncy': 'USDNOK', 'NZDUSDCR CMPN Curncy': 'USDNZD',
        'USDSEKCR CMPN Curncy': 'USDSEK',
    }
    currencies_factor = {
        'USDBRLCR CMPN Curncy': 1,  'USDMXNCR CMPN Curncy': 1,
        'USDCLPCR CMPN Curncy': 1,  'USDCOPCR CMPN Curncy': 1,
        'USDCNYCR CMPN Curncy': 1,  'USDINRCR CMPN Curncy': 1,
        'USDKRWCR CMPN Curncy': 1,  'USDPHPCR CMPN Curncy': 1,
        'USDSGDCR CMPN Curncy': 1,  'USDIDRCR CMPN Curncy': 1,
        'USDTHBCR CMPN Curncy': 1,  'USDTWDCR CMPN Curncy': 1,
        'USDCZKCR CMPN Curncy': 1,  'USDHUFCR CMPN Curncy': 1,
        'USDPLNCR CMPN Curncy': 1,  'USDZARCR CMPN Curncy': 1,
        'AUDUSDCR CMPN Curncy': -1, 'USDCADCR CMPN Curncy': 1,
        'EURUSDCR CMPN Curncy': -1, 'GBPUSDCR CMPN Curncy': -1,
        'USDCHFCR CMPN Curncy': 1,  'USDJPYCR CMPN Curncy': 1,
        'USDNOKCR CMPN Curncy': 1,  'NZDUSDCR CMPN Curncy': -1,
        'USDSEKCR CMPN Curncy': 1,
    }
    df = database.getData(tickers=currencies.keys()).pivot_table(
        index='date', columns='security', values='PX_LAST')
    for col, f in currencies_factor.items():
        df[col] = df[col] ** f
    df.rename(columns=currencies, inplace=True)
    return df


def get_fut_prices():
    bolsas = {
        'SPX Index': 'S&P',     'IBOV Index': 'Ibov',   'CF1 Index': 'CAC 40',
        'SM1 Index': 'Swiss',   'GX1 Index': 'DAX',     'VG1 Index': 'Stoxx',
        'XP1 Index': 'SPI 200', 'Z 1 Index': 'FTSE 100','NK1 Index': 'Nikkei',
        'NQ1 Index': 'NQ',      'KM1 Index': 'KOSPI',   'UX1 Index': 'VIX',
    }
    df = database.getData(tickers=bolsas.keys()).pivot_table(
        index='date', columns='security', values='PX_LAST')
    df.rename(columns=bolsas, inplace=True)
    return df


def get_comdty_prices_from_base():
    tickers = [
        'BO1 Comdty', 'SM1 Comdty', 'S 1 Comdty',  'W 1 Comdty',  'C 1 Comdty',
        'SB1 Comdty', 'KC1 Comdty', 'LH1 Comdty',  'LC1 Comdty',  'CC1 Comdty',
        'CT1 Comdty', 'FC1 Comdty', 'KW1 Comdty',  'O 1 Comdty',  'LB1 Comdty',
        'DA1 Comdty', 'JO1 Comdty', 'HO1 Comdty',  'NG1 Comdty',  'XB1 Comdty',
        'CO1 Comdty', 'CL1 Comdty', 'GC1 Comdty',  'SI1 Comdty',  'HG1 Comdty',
        'PA1 Comdty', 'PL1 Comdty', 'LA1 Comdty',  'LN1 Comdty',  'TY1 Comdty',
        'RX1 Comdty', 'IK1 Comdty', 'OAT1 Comdty', 'XM1 Comdty',  'JB1 Comdty',
        'CN1 Comdty', 'CKC1 COMB Comdty', 'XW1 Comdty',
    ]
    px = database.getData(tickers=tickers).pivot_table(
        index='date', columns='security', values='PX_LAST')
    meta = pd.read_excel('F:/Front/Moedas/Base/prices_futures.xlsx', sheet_name='Meta')
    meta = meta[meta['Class'].isin(['Agro', 'Metals', 'Energy', 'DM Rates'])]
    meta = meta[meta['Ticker'] != 'G 1 Comdty']
    rename_meta = dict(zip(meta['Ticker'], meta['Name.1']))
    return px.loc[:, list(rename_meta.keys())].rename(columns=rename_meta)


def get_comdty_meta():
    meta = pd.read_excel('F:/Front/Moedas/Base/prices_futures.xlsx', sheet_name='Meta')
    return meta[meta['Class'].isin(['Agro', 'Metals', 'Energy', 'DM Rates', 'Equity Index'])]


def generate_curncy_crosses(ccy_px):
    total, inverted = [], []
    for long in ccy_px.columns:
        for short in ccy_px.columns:
            if long[3:] != short[3:] and long[3:] + short[3:] not in inverted:
                total.append(1 / pd.DataFrame(
                    ccy_px[long].values / ccy_px[short].values,
                    columns=[long[3:] + short[3:]]))
                inverted.append(short[3:] + long[3:])
    df = pd.concat(total, axis=1)
    df.index = ccy_px.index
    return pd.concat([df, ccy_px], axis=1)


def get_dir_fx(fx_df):
    df = fx_df.copy().pow(-1)
    df.columns = fx_df.columns.str[3:]
    return df.assign(USD=1)


# ============================================================
# UNIVERSE METADATA & RETURNS
# ============================================================

fut_prices    = get_fut_prices()
ccy_prices    = get_ccy_prices()
comdty_prices = get_comdty_prices_from_base()
total_ccy_prices = generate_curncy_crosses(ccy_prices)

ccy_prices_dir    = get_dir_fx(ccy_prices)
ccy_rt_dir        = np.log(ccy_prices_dir.ffill() / ccy_prices_dir.ffill().shift(1))
ccy_rt_dir_stacked = ccy_rt_dir.stack().to_frame('ret').reset_index()

_DROP       = ['Lumber', 'USDPHP', 'Nickel', 'Aluminum', 'Orange Juice', 'Milk']
_DROP_DIR   = ['Lumber', 'PHP',    'Nickel', 'Aluminum', 'Orange Juice', 'Milk']

total_prices = pd.concat(
    [fut_prices.drop('VIX', axis=1), total_ccy_prices, comdty_prices], axis=1
).drop(_DROP, axis=1)

total_prices_dir = pd.concat(
    [fut_prices.drop('VIX', axis=1), ccy_prices_dir, comdty_prices], axis=1
).drop(_DROP_DIR, axis=1)

total_prices_dir_to_beta = pd.concat(
    [fut_prices, ccy_prices_dir, comdty_prices], axis=1
).drop(_DROP_DIR, axis=1)

total_fx_series = total_ccy_prices.columns.drop('USDPHP')

equity_cols    = fut_prices.columns
agro_futures   = ['Soybean Oil', 'Soybean Meal', 'Soybean', 'Wheat', 'Corn', 'Sugar',
                  'Coffee', 'Lean Hogs', 'Live Cattle', 'Cotton', 'Cattle Feeder',
                  'Cocoa', 'Oats', 'Winter Wheat']
energy_futures = ['Natural Gas', 'Gasoline', 'Brent', 'WTI', 'Heating Oil']
metals_futures = ['Gold', 'Silver', 'Copper', 'Palladium', 'Platinum']
dms_rates      = ['US Treasury 10y', 'Bund 10y', 'Italy 10y', 'France 10y',
                  'Australia 10y', 'Japan 10y', 'Canada 10y']
g10      = ['AUD', 'EUR', 'GBP', 'NZD', 'CAD', 'CHF', 'JPY', 'NOK', 'SEK']
ems      = ['BRL', 'CLP', 'COP', 'CZK', 'HUF', 'MXN', 'PLN', 'ZAR']
ems_asia = ['CNY', 'IDR', 'INR', 'KRW', 'SGD', 'THB', 'TWD']
latam    = ['BRL', 'CLP', 'COP', 'MXN', 'ZAR']
euro     = ['GBP', 'SEK', 'NOK', 'CHF', 'PLN', 'CZK', 'HUF']

cross_g10_series = total_prices.columns[
    ((total_prices.columns.str[3:].isin(g10)) & (total_prices.columns.str[:3].isin(g10))) |
    ((total_prices.columns.str[3:].isin(g10)) & total_prices.columns.str.contains('USD'))
]
cross_ems_series = total_prices.columns[
    ((total_prices.columns.str[3:].isin(ems)) & (total_prices.columns.str[:3].isin(ems))) |
    ((total_prices.columns.str[3:].isin(ems)) & total_prices.columns.str.contains('USD'))
]
cross_ems_asia_series = total_prices.columns[
    ((total_prices.columns.str[3:].isin(ems_asia)) & (total_prices.columns.str[:3].isin(ems_asia))) |
    ((total_prices.columns.str[3:].isin(ems_asia)) & total_prices.columns.str.contains('USD'))
]
cross_latamxusd_series = total_prices.columns[
    (total_prices.columns.str[3:].isin(latam)) & total_prices.columns.str.contains('USD')]
cross_g10xusd_series = total_prices.columns[
    (total_prices.columns.str[3:].isin(g10)) & total_prices.columns.str.contains('USD')]
cross_asiaxusd_series = total_prices.columns[
    (total_prices.columns.str[3:].isin(ems_asia)) & total_prices.columns.str.contains('USD')]

cross_groups    = ['CrossG10', 'CrossEMs', 'CrossEMsAsia']
groups          = ['Bolsas', 'Agro', 'Energy', 'Metals', 'Rates',
                   'CrossG10', 'CrossEMs', 'CrossEMsAsia']
cmdtys_groups   = ['Agro', 'Energy', 'Metals']
eqw_groups      = ['Bolsas', 'Cmdty', 'Rates', 'FX']
eqw_fx2x_groups = ['Bolsas', 'Cmdty', 'Rates', 'FX', 'FX2']
rp_groups       = ['Cmdty', 'RiskParity', 'FX']

sample_final = datetime.today().year
min_year_days = 5

total_returns     = np.log(total_prices.ffill() / total_prices.ffill().shift(1))
total_returns_dir = np.log(total_prices_dir.ffill() / total_prices_dir.ffill().shift(1))
total_returns_dir_to_beta = np.log(
    total_prices_dir_to_beta.ffill() / total_prices_dir_to_beta.ffill().shift(1))

total_returns_dir_ = total_returns_dir.copy()
total_returns_dir_.columns.name = 'asset'

def _apply_min_proxy(s): return np.sign(s).cumsum() >= min_year_days * 252 - 1

total_prices_start  = total_prices[total_prices.apply(_apply_min_proxy)].dropna(how='all')
total_prices_start  = total_prices_start[total_prices_start.index.year <= sample_final]
total_returns_start = np.log(total_prices_start.ffill() / total_prices_start.ffill().shift(1))
total_returns_start.columns.name = 'asset'
total_returns = total_returns[total_returns.index.year <= sample_final]

# ============================================================
# HELPERS
# ============================================================

class BacktestMetrics:
    @staticmethod
    def active_sharpe(s):
        a = s[s != 0].dropna()
        return a.mean() / a.std()
    @staticmethod
    def sharpe(s):
        return s.mean() / s.std()


def segregate_groups(pos_df):
    df = pos_df.copy()
    def _grp(a):
        if a in equity_cols:             return 'Bolsas'
        if a in agro_futures:            return 'Agro'
        if a in energy_futures:          return 'Energy'
        if a in metals_futures:          return 'Metals'
        if a in dms_rates:               return 'Rates'
        if a in cross_g10_series:        return 'G10'
        if a in cross_ems_series:        return 'EMs'
        if a in cross_ems_asia_series:   return 'EMsAsia'
        return np.nan
    def _cls(a):
        if a in equity_cols:                                            return 'Bolsas'
        if a in agro_futures + energy_futures + metals_futures:         return 'Comdty'
        if a in dms_rates:                                              return 'Rates'
        if a in (list(cross_g10_series) + list(cross_ems_series)
                 + list(cross_ems_asia_series)):                        return 'FX'
        return np.nan
    df['group']       = df['asset'].map(_grp)
    df['asset_class'] = df['asset'].map(_cls)
    return df


def plot_pnl(pnl_df, active_sharpe=True, vol_adj=False, return_fig=False, legend=True):
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(constrained_layout=True, figsize=(7, 11))
    fig.suptitle('Strategy Summary', fontsize=14, fontweight='bold')
    gs  = GridSpec(4, 4, figure=fig)
    ax1 = fig.add_subplot(gs[:2, :])
    ax2 = fig.add_subplot(gs[2:3, :])
    ax3 = fig.add_subplot(gs[3:, :])

    PNL = pnl_df[pnl_df != 0].replace([np.inf, -np.inf], np.nan).dropna(how='all')
    if vol_adj:
        PNL = PNL / PNL.std()
    PNL['Strat'] = PNL.sum(axis=1)
    PNL.drop('Strat', axis=1).cumsum().ffill().plot(ax=ax1)
    PNL[['Strat']].cumsum().ffill().plot(ax=ax2)
    (PNL[['Strat']].cumsum() - PNL[['Strat']].cumsum().cummax()).plot(ax=ax3)

    for ax, title in zip([ax1, ax2, ax3], ['PnL Composition', 'PnL Strategy', 'Drawdown']):
        ax.title.set_text(title)
        ax.title.set_fontweight('bold')
        ax.grid(axis='both')
        ax.margins(x=0.05)
        ax.autoscale()

    SHARPE = PNL.apply(
        lambda s: np.sqrt(252) * (s[s != 0].dropna().mean() / s[s != 0].dropna().std())
        if active_sharpe else np.sqrt(252) * s.dropna().mean() / s.dropna().std()
    )
    if legend:
        ax1.legend(
            labels=[f"{c}={SHARPE.loc[c].round(2)}" for c in SHARPE.index if c != 'Strat'],
            ncols=2)
        ax2.legend(labels=[f"Strat={SHARPE.loc['Strat'].round(2)}"])
    else:
        ax1.legend(labels=[None])

    if return_fig:
        return fig
    return PNL

# ============================================================
# SIGNAL CALCULATION
# ============================================================

def calculate_performance(
    signal_type, signal, holding_period,
    returns_df=None, returns_vol_df=None,
    lag=0, fill_lag=False, start=False, until=2019,
    thresh=0.0, norm_wdw=252 * 2, return_holding_period=False,
    return_pnl=False, size_smoothing=False, add_cost=True,
    momentum=False, momentum_roll=False, inverse=False, norm='vol',
):
    if returns_df is None:     returns_df     = total_returns_start
    if returns_vol_df is None: returns_vol_df = total_returns

    sig = signal.copy().shift(lag)
    sig = sig[sig.columns[~sig.columns.str.contains('PHP')]]
    sig = sig[sig.columns[sig.columns != 'UK 10y']]

    sig[np.abs(sig) < thresh] = np.nan
    position_final = sig.applymap(np.sign)
    position_final = position_final[position_final != 0]

    roll_window = holding_period - lag if fill_lag else holding_period
    pos_hold = position_final.fillna(0).rolling(roll_window).mean()
    if not size_smoothing:
        pos_hold = pos_hold.applymap(np.sign)

    if norm == 'vol':
        norm_factor = returns_df.rolling(norm_wdw, min_periods=norm_wdw).std().shift(2)
    elif norm == 'var':
        norm_factor = returns_df.rolling(norm_wdw, min_periods=norm_wdw).quantile(0.025).abs().shift(2)
    elif norm == 'abs':
        norm_factor = returns_df.abs().rolling(norm_wdw, min_periods=norm_wdw).quantile(0.95).shift(2)

    returns_to_pnl = returns_df / norm_factor
    pos_final = pos_hold.loc[:, ~pos_hold.columns.duplicated()]
    pnl = pos_final.mul(returns_to_pnl.loc[:, ~returns_df.columns.duplicated()])

    pos_norm = pos_hold / norm_factor
    pos_norm.columns.name = 'asset'
    pos_norm.index.name   = 'date'

    if until:  pnl = pnl.loc[pnl.index.year <= until].copy()
    if start:  pnl = pnl.loc[pnl.index.year >= start].copy()

    if add_cost:
        cost = pos_hold.loc[:, ~pos_hold.columns.duplicated()].diff().applymap(np.abs)
        pnl -= cost.mul(norm_factor * 0.05)

    pnl['CrossG10']     = pnl[cross_g10_series].mean(axis=1)
    pnl['CrossEMs']     = pnl[cross_ems_series].mean(axis=1)
    pnl['CrossEMsAsia'] = pnl[cross_ems_asia_series].mean(axis=1)
    pnl['CrossLatam']   = pnl[cross_latamxusd_series].mean(axis=1)
    pnl['AnyFX']        = pnl[total_fx_series].mean(axis=1)
    pnl['Bolsas']       = pnl[equity_cols].mean(axis=1)
    pnl['Agro']         = pnl[agro_futures].mean(axis=1)
    pnl['Energy']       = pnl[energy_futures].mean(axis=1)
    pnl['Metals']       = pnl[metals_futures].mean(axis=1)
    pnl['Rates']        = pnl[dms_rates].mean(axis=1)
    pnl['FX']           = pnl[cross_groups].mean(axis=1)
    pnl['FX2']          = pnl[cross_groups].mean(axis=1)
    pnl['Cmdty']        = pnl[cmdtys_groups].mean(axis=1)
    pnl['RiskParity']   = (pnl[equity_cols].mean(axis=1) + pnl[dms_rates].mean(axis=1))
    pnl['Agg']          = pnl[groups].mean(axis=1)
    pnl['AggEQW']       = pnl[eqw_groups].mean(axis=1)
    pnl['AggEQW_FX2x']  = pnl[eqw_fx2x_groups].mean(axis=1)
    pnl['EQW']          = pnl[returns_df.columns].mean(axis=1)
    pnl['AggRiskParity'] = pnl[rp_groups].mean(axis=1)

    if return_holding_period:
        return pnl[pnl != 0].count(numeric_only=True) / pnl.count()

    if return_pnl:
        return (pnl[pnl != 0].replace(np.inf, 0),
                pos_hold[pos_hold != 0],
                pos_norm[pos_norm != 0])

    return np.sqrt(252) * pnl.apply(BacktestMetrics.sharpe)

# ============================================================
# PIPELINE HELPERS
# ============================================================

VAR_TARGET   = -100
VAR_WINDOW   = 2 * 252
VAR_QUANTILE = 0.025
MIN_OBS      = 2 * 252

SIMPLE_CLASS_WEIGHTS = {
    'Agro': 0.33, 'Energy': 0.33, 'Metals': 0.33,
    'Bolsas': 1.0, 'Rates': 1.0,
    'G10': 0.33, 'EMs': 0.33, 'EMsAsia': 0.33,
}
FINAL_WEIGHTS = {'Comdty': 1, 'Bolsas': 1, 'Rates': 1, 'FX': 1}

BENCHMARK_TICKERS = {
    'sp500':    'S&P',
    'treasury': 'US Treasury 10y',
    'cnh':      'CNY',
    'oil':      'WTI',
    'gold':     'Gold',
    'eur':      'EUR',
    'vix':      'VIX',
    'corn':  'Corn'
}


def _var_adjust_series(s):
    """Scale series so its historical VaR matches VAR_TARGET."""
    return s * (VAR_TARGET / s.rolling(VAR_WINDOW, min_periods=252).quantile(VAR_QUANTILE).shift(1))


def adjust_pnl_pos_to_dash(pnl, pos, pos_adj):
    """Stack and annotate with group/asset_class. Used for position display tables."""
    def _stack(df, col):
        s = df.stack(dropna=False).reset_index()
        s.columns = ['date', 'asset', col]
        return segregate_groups(s)
    return _stack(pnl, 'pnl'), _stack(pos, 'position'), _stack(pos_adj, 'pos_adj')

# ── SHAP helpers ──────────────────────────────────────────────────────────

# Maps panel name → list of asset names as they appear in fin_pos_asset.
# FX assets are individual currency codes after cross decomposition.
SHAP_GROUP_ASSETS = {
    'Equity':     None,          # resolved at runtime using equity_cols
    'Rates':      dms_rates,
    'Energy':     energy_futures,
    'Metals':     metals_futures,
    'Agro':       agro_futures,
    'FX G10':     g10,
    'FX EMs':     ems,
    'FX EMsAsia': ems_asia,
}


def _rescaled_sharpe(pnl_series, var_target=VAR_TARGET,
                     window=VAR_WINDOW, ann=252):
    """Annualised Sharpe after rolling-VaR rescaling to var_target."""
    var = (pnl_series
           .rolling(window, min_periods=252)
           .quantile(VAR_QUANTILE)
           .shift(1))
    rescaled = (pnl_series * (var_target / var)).dropna()
    return np.sqrt(ann) * rescaled.mean() / rescaled.std()


def compute_shap_sharpe(fin_pos_asset):
    """
    SHAP Sharpe by group:
        SHAP(g) = Sharpe(full_rescaled) - Sharpe(without_g_rescaled)

    fin_pos_asset : date x asset DataFrame of final notional positions.
    Returns       : (pd.Series of SHAP values indexed by group, float full_sharpe)
    """
    group_map = {**SHAP_GROUP_ASSETS, 'Equity': list(equity_cols)}

    ret    = total_returns_dir_.reindex(columns=fin_pos_asset.columns)
    common = fin_pos_asset.index.intersection(ret.index)
    pos    = fin_pos_asset.reindex(common).ffill()
    ret    = ret.reindex(common)

    pnl_full = pos.mul(ret).sum(axis=1)
    sh_full  = _rescaled_sharpe(pnl_full)

    shap = {}
    for grp, assets in group_map.items():
        valid    = [a for a in assets if a in pos.columns]
        pos_sub  = pos.copy()
        pos_sub[valid] = 0.0
        pnl_sub  = pos_sub.mul(ret).sum(axis=1)
        shap[grp] = sh_full - _rescaled_sharpe(pnl_sub)

    return pd.Series(shap).round(3), round(sh_full, 3)


# ============================================================
# BASE PIPELINE  (heavy — run once per signal)
# ============================================================

def run_base_pipeline(pnl_raw, pos_raw, pos_vol_raw):
    """
    Full pipeline: FX decomp → rolling group VaR → portfolio VaR → beta computation.
    div_factor NOT applied here. Call apply_beta_hedging() for the hedged PnL.

    Returns
    -------
    dict with keys:
        pnl_grouped, pos_grouped, pos_adj_grouped  : for position display
        pnl_raw_groups                             : pnl[groups] — unweighted class PnL
        pnl_dir_var_grp                            : group-level VaR-rescaled PnL
        pnl_final_total                            : portfolio-level VaR-rescaled PnL
        portfolio_beta                             : raw rolling betas (pre-div)
        portfolio_beta_exp                         : normalised betas (pre-div)
        benchmark_series                           : dict of VaR-adj benchmark returns
    """
    # ── 0. Grouped stack for dashboard tables ────────────────────────────
    pnl_grouped, pos_grouped, pos_adj_grouped = adjust_pnl_pos_to_dash(
        pnl_raw, pos_raw, pos_vol_raw)

    # ── 1. Position construction (vol-normalised) ─────────────────────────
    pos_stacked = (pos_vol_raw
                   .rename_axis(index='date', columns='asset')
                   .stack().to_frame('ind_pos').reset_index())
    pos_cls = segregate_groups(pos_stacked).dropna()
    pos_cls = pos_cls.merge(total_returns_start.stack().to_frame('ret'),
                            on=['date', 'asset'])

    grp = pos_cls.groupby(['date', 'group'])
    pos_cls['group_assets'] = grp['asset'].transform(lambda x: x.nunique())
    pos_cls['usd_assets']   = grp['asset'].transform(
        lambda x: sum('USD' in c for c in x.unique()))
    pos_cls['bsk_pos'] = pos_cls['ind_pos'] / pos_cls['group_assets']

    # ── 2. FX cross decomposition ─────────────────────────────────────────
    not_fx = pos_cls[pos_cls['asset_class'] != 'FX']
    fx     = pos_cls[pos_cls['asset_class'] == 'FX'].copy()
    fx['long']      =  fx['asset'].str[:3]
    fx['short']     =  fx['asset'].str[3:]
    fx['pos_long']  =  fx['bsk_pos']
    fx['pos_short'] = -fx['bsk_pos']

    fx_full = (fx
               .merge(ccy_rt_dir_stacked.rename(columns={'security': 'long'}),
                      on=['date', 'long'],  suffixes=('', '_long'))
               .merge(ccy_rt_dir_stacked.rename(columns={'security': 'short'}),
                      on=['date', 'short'], suffixes=('', '_short')))

    pos_long  = (fx_full
                 .groupby(['date', 'long',  'group', 'asset_class'], as_index=False)
                 ['pos_long'].sum().rename(columns={'long': 'asset'}))
    pos_short = (fx_full
                 .groupby(['date', 'short', 'group', 'asset_class'], as_index=False)
                 ['pos_short'].sum().rename(columns={'short': 'asset'}))
    pos_fx = pos_long.merge(pos_short, on=['date', 'asset', 'asset_class', 'group'], how='outer')
    pos_fx['bsk_pos'] = pos_fx['pos_long'].fillna(0) + pos_fx['pos_short'].fillna(0)
    pos_fx = pos_fx.drop(columns=['pos_long', 'pos_short'])

    COLS = pos_fx.columns.tolist()
    position_features = (pd.concat([pos_fx, not_fx[COLS]])
                         .sort_values('date').reset_index(drop=True))

    # ── 3. Rolling VaR per group ──────────────────────────────────────────
    var_ls = []
    for gr, df_gr in position_features.groupby('group'):
        gr_pos = df_gr.pivot_table(
            index='date', columns='asset', values='bsk_pos', aggfunc='sum')
        records = []
        for df_pos in gr_pos.rolling(1):
            date     = df_pos.index[-1]
            hist_ret = total_returns_dir.shift(2).loc[:date].tail(VAR_WINDOW)
            if hist_ret.dropna(how='all').size < MIN_OBS:
                records.append({'date': date, 'VaR': float('nan')})
                continue
            port_pnl = hist_ret.mul(df_pos.iloc[-1], axis=1).sum(axis=1)
            records.append({'date': date, 'VaR': port_pnl.quantile(VAR_QUANTILE)})
        sim_var = pd.DataFrame(records).set_index('date')
        var_ls.append(df_gr.merge(sim_var.reset_index(), on='date'))

    pos_var = pd.concat(var_ls, ignore_index=True)
    pos_var['VaR_Adjust'] = VAR_TARGET / pos_var['VaR']
    pos_var['VaR_pos']    = pos_var['bsk_pos'] * pos_var['VaR_Adjust']

    # ── 4. Merge returns and apply group/class weights ────────────────────
    pos_var_ret = pos_var.merge(
        total_returns_dir_.stack().to_frame('ret_dir'), on=['date', 'asset'])
    pos_var_ret['weight_cls']    = pos_var_ret['group'].map(SIMPLE_CLASS_WEIGHTS)
    pos_var_ret['weight_strat']  = pos_var_ret['asset_class'].map(FINAL_WEIGHTS)
    pos_var_ret['pnl_final_grp'] = pos_var_ret['VaR_pos'] * pos_var_ret['ret_dir']
    pos_var_ret['pnl_final']     = pos_var_ret['pnl_final_grp'] * pos_var_ret['weight_strat']

    pnl_dir_var_grp = pos_var_ret.pivot_table(
        index='date', columns='group',
        values='pnl_final_grp', aggfunc='sum', fill_value=0)

    # ── 5. Portfolio VaR rescaling ────────────────────────────────────────
    portfolio_pos = (pos_var_ret
                     .assign(final_pos=lambda x: x['VaR_pos'] * x['weight_strat'])
                     .groupby(['date', 'asset'])['final_pos']
                     .sum().unstack('asset'))

    bm_series = {k: _var_adjust_series(total_returns_dir_to_beta[v])
                 for k, v in BENCHMARK_TICKERS.items()}

    corr_records = []
    for df_pos in portfolio_pos.rolling(1):
        date     = df_pos.index[-1]
        hist_ret = total_returns_dir.shift(2).loc[:date].tail(VAR_WINDOW)
        if hist_ret.dropna(how='all').size < MIN_OBS:
            corr_records.append({
                'date': date, 'VaR_portfolio': float('nan'),
                **{f'beta_{k}': float('nan') for k in bm_series},
            })
            continue
        port_pnl = hist_ret.mul(df_pos.iloc[-1], axis=1).sum(axis=1)
        idx = port_pnl.index
        row = {'date': date, 'VaR_portfolio': port_pnl.quantile(VAR_QUANTILE)}
        for k, bm in bm_series.items():
            bm_w = bm.shift(2).reindex(idx)
            row[f'beta_{k}'] = port_pnl.cov(bm_w) / bm_w.var()
        corr_records.append(row)

    portfolio_stats = pd.DataFrame(corr_records).set_index('date')
    portfolio_beta  = (portfolio_stats
                       .drop(columns=['VaR_portfolio'])
                       .round(2).fillna(0))

    # Normalised betas (row-sum-to-1 in abs, rounded to 0.05)
    portfolio_beta_exp = (portfolio_beta
                          .div(portfolio_beta.abs().sum(axis=1), axis=0)
                          .div(0.05).round().mul(0.05))

    # Portfolio-level VaR rescale
    port_var_adj = (VAR_TARGET / portfolio_stats['VaR_portfolio']).rename('pva')
    pos_var_ret['pva']           = pos_var_ret['date'].map(port_var_adj)
    pos_var_ret['VaR_pos_total'] = pos_var_ret['VaR_pos'] * pos_var_ret['pva']
    pos_var_ret['pnl_final_total'] = (pos_var_ret['VaR_pos_total']
                                      * pos_var_ret['weight_strat']
                                      * pos_var_ret['ret_dir'])

    pnl_final_total = pos_var_ret.pivot_table(
        index='date', columns='asset_class',
        values='pnl_final_total', aggfunc='sum', fill_value=0)

    # ── Financial positions (VaR-target units, weight_strat applied) ──────
    pos_var_ret['fin_pos'] = pos_var_ret['VaR_pos_total'] * pos_var_ret['weight_strat']

    fin_pos_cls = (pos_var_ret
                   .groupby(['date', 'asset_class'])['fin_pos']
                   .sum().unstack('asset_class'))

    # Group-level breakdown (Bolsas, Agro, Energy, Metals, Rates, G10, EMs, EMsAsia)
    fin_pos_group = (pos_var_ret
                     .groupby(['date', 'group'])['fin_pos']
                     .sum().unstack('group'))

    # Per-asset final notional — FX assets are individual currency codes after decomp
    fin_pos_asset = (pos_var_ret
                     .groupby(['date', 'asset'])['fin_pos']
                     .sum().unstack('asset'))

    # SHAP Sharpe by group
    shap_sharpe, sharpe_full = compute_shap_sharpe(fin_pos_asset)

    return {
        'pnl_grouped':        pnl_grouped,
        'pos_grouped':        pos_grouped,
        'pos_adj_grouped':    pos_adj_grouped,
        # Raw position inputs are kept only for dashboard forward projection.
        # They do not change any historical PnL, signal, hedge or sizing logic.
        'pos_raw_input':      pos_raw,
        'pos_vol_raw_input':  pos_vol_raw,
        'pnl_raw_groups':     pnl_raw,
        'pnl_dir_var_grp':    pnl_dir_var_grp,
        'pnl_final_total':    pnl_final_total,
        'portfolio_beta':     portfolio_beta,
        'portfolio_beta_exp': portfolio_beta_exp,
        'benchmark_series':   bm_series,
        'fin_pos_cls':        fin_pos_cls,    # date × asset_class
        'fin_pos_group':      fin_pos_group,  # date × group
        'fin_pos_asset':      fin_pos_asset,  # date × asset (granular)
        'shap_sharpe':        shap_sharpe,    # Series: group → Sharpe contribution
        'sharpe_full':        sharpe_full,    # float: full portfolio Sharpe
    }


# ============================================================
# BETA HEDGING  (light — reactive to div_factor slider)
# ============================================================

def apply_beta_hedging(base, div_factor=2.0):
    """
    Applies beta hedging using pre-computed base pipeline results.
    Only arithmetic operations — fast enough to call on every slider change.

    Parameters
    ----------
    base       : dict returned by run_base_pipeline()
    div_factor : float in [1, 3], step 0.5

    Returns
    -------
    dict with:
        portfolio_beta_exp_  : beta exposure table (post-div)
        pnl_scenarios        : DataFrame with hedged/unhedged PnL columns
        pnl_hedges           : DataFrame with individual hedge PnL columns
    """
    pnl_base = base['pnl_final_total'].sum(axis=1).rename('no_hedge')
    bm       = base['benchmark_series']

    # Individual betas (raw, divided by div_factor)
    pb_ind = base['portfolio_beta'] / div_factor
    # Normalised betas (for full hedge — preserves relative weights)
    pb_exp = base['portfolio_beta_exp'] / div_factor

    def _hedge(beta_col, bm_key):
        return -pb_ind[beta_col] * bm[bm_key]

    def _hedge_exp(beta_col, bm_key):
        return -pb_exp[beta_col] * bm[bm_key]

    individual_hedges = {k: _hedge(f'beta_{k}', k) for k in bm}
    full_hedge = sum(_hedge_exp(f'beta_{k}', k) for k in bm).fillna(0)

    # Notional of each individual beta hedge leg (in same VaR-target units as fin_pos_cls)
    # -pb_ind[beta_k] = units of that benchmark to short to offset the exposure.
    beta_hedge_notional_per_bm = pd.DataFrame(
        {k: -pb_ind[f'beta_{k}'] for k in bm}
    )
    # Single "full hedge" notional series — sum of all individual legs
    beta_hedge_full = beta_hedge_notional_per_bm.sum(axis=1).rename('hedge_full')

    pnl_scenarios = pd.DataFrame({
        'no_hedged':    pnl_base,
        **{f'hedged_{k}': pnl_base + h for k, h in individual_hedges.items()},
        'hedged_full':  pnl_base + full_hedge,
    }).dropna(how='all')

    pnl_hedges = pd.DataFrame({
        **{f'hedge_{k}': h for k, h in individual_hedges.items()},
        'hedge_full': full_hedge,
    }).dropna(how='all')

    return {
        'portfolio_beta_exp_':         pb_exp,
        'pnl_scenarios':               pnl_scenarios,
        'pnl_hedges':                  pnl_hedges,
        'beta_hedge_notional_per_bm':  beta_hedge_notional_per_bm,  # date × benchmark
        'beta_hedge_full':             beta_hedge_full,              # date Series (full hedge notional)
    }


# ============================================================
# FORWARD POSITION PROJECTION FOR DASHBOARD TABLES ONLY
# ============================================================

FORWARD_POSITION_DAYS = 5
FORWARD_POSITION_FREQ = 'B'  # 'B' = today plus next business days


def _normalise_datetime_index(df):
    """Return a sorted copy with a normalized DatetimeIndex."""
    out = df.copy()
    out.index = pd.to_datetime(out.index).normalize()
    return out[~out.index.duplicated(keep='last')].sort_index()


def _last_row_on_or_before(df, date_ref):
    """Last non-empty row on or before date_ref as a one-row DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame()
    x = _normalise_datetime_index(df)
    ref = pd.Timestamp(date_ref).normalize()
    x = x.loc[x.index <= ref].dropna(how='all')
    if x.empty:
        return pd.DataFrame()
    return x.tail(1)


def _forward_position_dates(date_ref, days=FORWARD_POSITION_DAYS,
                            freq=FORWARD_POSITION_FREQ):
    """
    Dates displayed in the dashboard position tables.

    For business-day projection, the first row is always date_ref and the
    remaining rows are the next `days` business days. This keeps "today" in
    the table even when the selected reference date is not a business day.
    """
    start = pd.Timestamp(date_ref).normalize()
    if str(freq).upper() in ['B', 'BUSINESS', 'BUSINESS_DAY']:
        next_days = pd.bdate_range(start=start + pd.offsets.BDay(1), periods=days)
        return pd.DatetimeIndex([start]).append(next_days)
    return pd.date_range(start=start, periods=days + 1, freq=freq)


def _current_vol_scale_for_projection(pos_raw, pos_vol_raw, date_ref,
                                      norm_wdw=252 * 2):
    """
    Current vol scale used to infer future vol-normalised positions.

    The historical strategy already calculates pos_vol_raw as pos_raw divided
    by the rolling vol. For forward dates the future rolling vol is unknown, so
    this helper freezes the latest available vol scale as of date_ref.
    """
    pos_raw = _normalise_datetime_index(pos_raw)
    pos_vol_raw = _normalise_datetime_index(pos_vol_raw)
    cols = pos_raw.columns
    ref = pd.Timestamp(date_ref).normalize()

    # Primary source: the same rolling-vol definition used by calculate_performance(norm='vol').
    norm_factor = (total_returns_start.reindex(columns=cols)
                   .rolling(norm_wdw, min_periods=norm_wdw)
                   .std()
                   .shift(2))
    scale_from_vol = (1.0 / norm_factor.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    snap_vol = _last_row_on_or_before(scale_from_vol, ref)

    # Fallback source: infer the scale from the already calculated current position tables.
    raw_nonzero = pos_raw.replace(0, np.nan)
    scale_from_output = (pos_vol_raw.divide(raw_nonzero)
                         .replace([np.inf, -np.inf], np.nan))
    snap_output = _last_row_on_or_before(scale_from_output.ffill(), ref)

    if snap_vol.empty and snap_output.empty:
        return pd.Series(index=cols, dtype=float)
    if snap_vol.empty:
        return snap_output.iloc[-1].reindex(cols)
    if snap_output.empty:
        return snap_vol.iloc[-1].reindex(cols)
    return snap_vol.iloc[-1].reindex(cols).combine_first(snap_output.iloc[-1].reindex(cols))


def _project_pos_vol_with_current_vol(pos_raw, pos_vol_raw, date_ref,
                                      days=FORWARD_POSITION_DAYS,
                                      freq=FORWARD_POSITION_FREQ):
    """
    Build future vol-normalised raw positions for table display only.

    Future raw holdings come from the existing pos_raw calendar when available.
    If a projected date is not present in pos_raw, the latest known row is used.
    NaNs on existing dates are preserved, so an explicit zero/no-position date
    is not accidentally forward-filled.
    """
    pos_raw = _normalise_datetime_index(pos_raw)
    dates = _forward_position_dates(date_ref, days=days, freq=freq)
    scale = _current_vol_scale_for_projection(pos_raw, pos_vol_raw, date_ref)

    raw_window = pos_raw.reindex(dates, method='ffill')
    pos_vol_projection = raw_window.mul(scale, axis=1)
    pos_vol_projection.index.name = 'date'
    pos_vol_projection.columns.name = 'asset'
    return pos_vol_projection, dates


def _position_features_from_pos_vol(pos_vol_projection):
    """Replicate the position construction used by run_base_pipeline, without future returns."""
    pos_stacked = (pos_vol_projection
                   .rename_axis(index='date', columns='asset')
                   .stack(dropna=True)
                   .to_frame('ind_pos')
                   .reset_index())
    if pos_stacked.empty:
        return pd.DataFrame(columns=['date', 'asset', 'group', 'asset_class', 'bsk_pos'])

    pos_cls = segregate_groups(pos_stacked).dropna(subset=['group', 'asset_class'])
    pos_cls = pos_cls.replace([np.inf, -np.inf], np.nan).dropna(subset=['ind_pos'])
    pos_cls = pos_cls[pos_cls['ind_pos'] != 0]
    pos_cls = pos_cls[pos_cls['asset'].isin(total_returns_start.columns)]
    if pos_cls.empty:
        return pd.DataFrame(columns=['date', 'asset', 'group', 'asset_class', 'bsk_pos'])

    grp = pos_cls.groupby(['date', 'group'])
    pos_cls['group_assets'] = grp['asset'].transform(lambda x: x.nunique())
    pos_cls['bsk_pos'] = pos_cls['ind_pos'] / pos_cls['group_assets']

    feature_cols = ['date', 'asset', 'group', 'asset_class', 'bsk_pos']
    not_fx = pos_cls[pos_cls['asset_class'] != 'FX'][feature_cols]

    fx = pos_cls[pos_cls['asset_class'] == 'FX'].copy()
    fx_frames = []
    if not fx.empty:
        fx['long'] = fx['asset'].str[:3]
        fx['short'] = fx['asset'].str[3:]
        long_leg = (fx[['date', 'long', 'group', 'asset_class', 'bsk_pos']]
                    .rename(columns={'long': 'asset'}))
        short_leg = (fx[['date', 'short', 'group', 'asset_class', 'bsk_pos']]
                     .rename(columns={'short': 'asset'}))
        short_leg['bsk_pos'] = -short_leg['bsk_pos']
        pos_fx = pd.concat([long_leg, short_leg], ignore_index=True)
        pos_fx = (pos_fx
                  .groupby(['date', 'asset', 'group', 'asset_class'], as_index=False)
                  ['bsk_pos'].sum())
        pos_fx = pos_fx[pos_fx['asset'].isin(total_returns_dir.columns)]
        fx_frames.append(pos_fx[feature_cols])

    frames = [not_fx] + fx_frames
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=feature_cols)
    return pd.concat(frames, ignore_index=True).sort_values('date').reset_index(drop=True)


def _hist_returns_for_projection(date_ref):
    """Historical return window available as of date_ref, matching the strategy lag convention."""
    ref = pd.Timestamp(date_ref).normalize()
    hist_ret = total_returns_dir.shift(2).loc[:ref].tail(VAR_WINDOW)
    return hist_ret


def _simulate_forward_fin_positions(pos_vol_projection, date_ref):
    """
    Convert projected vol-normalised positions into final dashboard notionals.

    This mirrors the existing VaR and beta arithmetic for display only. It does
    not feed back into BASE_RESULTS, historical PnL, SHAP, signals or hedging.
    """
    dates = pd.DatetimeIndex(pos_vol_projection.index)
    position_features = _position_features_from_pos_vol(pos_vol_projection)
    beta_cols = [f'beta_{k}' for k in BENCHMARK_TICKERS]

    empty_fin = pd.DataFrame(index=dates)
    empty_beta = pd.DataFrame(index=dates, columns=beta_cols, dtype=float)
    if position_features.empty:
        return {'fin_pos_asset': empty_fin, 'portfolio_beta': empty_beta.fillna(0)}

    hist_ret = _hist_returns_for_projection(date_ref)
    if hist_ret.dropna(how='all').size < MIN_OBS:
        return {'fin_pos_asset': empty_fin, 'portfolio_beta': empty_beta.fillna(0)}

    var_ls = []
    for gr, df_gr in position_features.groupby('group'):
        gr_pos = df_gr.pivot_table(index='date', columns='asset', values='bsk_pos', aggfunc='sum')
        records = []
        for dt, row in gr_pos.iterrows():
            pos = row.fillna(0)
            port_pnl = hist_ret.mul(pos, axis=1).sum(axis=1)
            records.append({'date': dt, 'group': gr, 'VaR': port_pnl.quantile(VAR_QUANTILE)})
        sim_var = pd.DataFrame(records)
        var_ls.append(df_gr.merge(sim_var, on=['date', 'group'], how='left'))

    pos_var = pd.concat(var_ls, ignore_index=True)
    pos_var['VaR_Adjust'] = VAR_TARGET / pos_var['VaR'].replace(0, np.nan)
    pos_var['VaR_pos'] = pos_var['bsk_pos'] * pos_var['VaR_Adjust']
    pos_var['weight_strat'] = pos_var['asset_class'].map(FINAL_WEIGHTS)
    pos_var = pos_var.replace([np.inf, -np.inf], np.nan).dropna(subset=['VaR_pos', 'weight_strat'])

    if pos_var.empty:
        return {'fin_pos_asset': empty_fin, 'portfolio_beta': empty_beta.fillna(0)}

    portfolio_pos = (pos_var
                     .assign(final_pos=lambda x: x['VaR_pos'] * x['weight_strat'])
                     .groupby(['date', 'asset'])['final_pos']
                     .sum()
                     .unstack('asset'))

    bm_series = {k: _var_adjust_series(total_returns_dir_to_beta[v])
                 for k, v in BENCHMARK_TICKERS.items()}

    stats_records = []
    for dt, row in portfolio_pos.iterrows():
        pos = row.fillna(0)
        port_pnl = hist_ret.mul(pos, axis=1).sum(axis=1)
        row_stats = {'date': dt, 'VaR_portfolio': port_pnl.quantile(VAR_QUANTILE)}
        for k, bm in bm_series.items():
            bm_w = bm.shift(2).reindex(hist_ret.index)
            var_bm = bm_w.var()
            row_stats[f'beta_{k}'] = port_pnl.cov(bm_w) / var_bm if var_bm != 0 else np.nan
        stats_records.append(row_stats)

    portfolio_stats = pd.DataFrame(stats_records).set_index('date')
    portfolio_beta = (portfolio_stats
                      .drop(columns=['VaR_portfolio'])
                      .round(2)
                      .fillna(0)
                      .reindex(dates))

    port_var_adj = (VAR_TARGET / portfolio_stats['VaR_portfolio'].replace(0, np.nan)).rename('pva')
    pos_var['pva'] = pos_var['date'].map(port_var_adj)
    pos_var['fin_pos'] = pos_var['VaR_pos'] * pos_var['pva'] * pos_var['weight_strat']
    pos_var = pos_var.replace([np.inf, -np.inf], np.nan).dropna(subset=['fin_pos'])

    fin_pos_asset = (pos_var
                     .groupby(['date', 'asset'])['fin_pos']
                     .sum()
                     .unstack('asset')
                     .reindex(dates))

    return {'fin_pos_asset': fin_pos_asset, 'portfolio_beta': portfolio_beta}


def build_forward_position_projection(base, date_ref, days=FORWARD_POSITION_DAYS,
                                      freq=FORWARD_POSITION_FREQ):
    """
    Dashboard-only projection from date_ref through the next `days` days.

    The first row is forced to match the already-computed current strategy
    position. Future rows use the existing raw position calendar with the latest
    available vol scale frozen as of date_ref.
    """
    pos_raw = base['pos_raw_input']
    pos_vol_raw = base['pos_vol_raw_input']
    pos_vol_projection, dates = _project_pos_vol_with_current_vol(
        pos_raw, pos_vol_raw, date_ref, days=days, freq=freq)

    proj = _simulate_forward_fin_positions(pos_vol_projection, date_ref)
    fin_pos_asset = proj['fin_pos_asset'].copy()
    portfolio_beta = proj['portfolio_beta'].copy()

    current_fin = _last_row_on_or_before(base['fin_pos_asset'], date_ref)
    if not current_fin.empty and len(dates) > 0:
        all_cols = fin_pos_asset.columns.union(current_fin.columns)
        fin_pos_asset = fin_pos_asset.reindex(index=dates, columns=all_cols)
        fin_pos_asset.loc[dates[0], current_fin.columns] = current_fin.iloc[-1]

    current_beta = _last_row_on_or_before(base['portfolio_beta'], date_ref)
    if not current_beta.empty and len(dates) > 0:
        all_cols = portfolio_beta.columns.union(current_beta.columns)
        portfolio_beta = portfolio_beta.reindex(index=dates, columns=all_cols)
        portfolio_beta.loc[dates[0], current_beta.columns] = current_beta.iloc[-1]
    portfolio_beta = portfolio_beta.fillna(0)

    return {
        'dates': dates,
        'pos_vol_projection': pos_vol_projection,
        'fin_pos_asset': fin_pos_asset,
        'portfolio_beta': portfolio_beta,
    }


# ============================================================
# SIGNAL COMPUTATION
# ============================================================

season_h3_exp_Y_hit    = pd.read_excel('F:/Front/Moedas/Working/Sazonalidade/Export/season_hitratio_h3_exp_Y.xlsx',       index_col=0)
season_h5_exp_Y_hit    = pd.read_excel('F:/Front/Moedas/Working/Sazonalidade/Export/season_hitratio_h5_exp_Y.xlsx',       index_col=0)
season_h5_exp_Y_tstat  = pd.read_excel('F:/Front/Moedas/Working/Sazonalidade/Export/season_tstat_h5_exp_Y.xlsx',          index_col=0)
season_h3_exp_Y_tr_hit = pd.read_excel('F:/Front/Moedas/Working/Sazonalidade/Export/season_trade_hitratio_h3_exp_Y.xlsx', index_col=0)

pnl_exp_full_hit, pos_exp_full_hit, pos_exp_full_vol_hit = calculate_performance(
    signal_type='hitratio', signal=season_h5_exp_Y_hit,
    holding_period=5, thresh=0.075, lag=-2, size_smoothing=True,
    until=sample_final, return_pnl=True, norm='vol')

pnl_exp_full_tr_hit, pos_exp_full_tr_hit, pos_exp_full_vol_tr_hit = calculate_performance(
    signal_type='hitratio', signal=season_h3_exp_Y_hit,
    holding_period=3, thresh=0.075, lag=-1, size_smoothing=True,
    until=sample_final, return_pnl=True, norm='vol')

pnl_exp_full_tstat, pos_exp_full_tstat, pos_exp_full_vol_tstat = calculate_performance(
    signal_type='tstat', signal=season_h5_exp_Y_tstat,
    holding_period=5, thresh=1.96, lag=-2, size_smoothing=True,
    until=sample_final, return_pnl=True, norm='vol')

# ============================================================
# PRE-COMPUTE BASE PIPELINES  (run once on import)
# ============================================================

SIGNAL_DEFS = {
    'hitratio':     (pnl_exp_full_hit,    pos_exp_full_hit,    pos_exp_full_vol_hit),
    'trd_hitratio': (pnl_exp_full_tr_hit, pos_exp_full_tr_hit, pos_exp_full_vol_tr_hit),
    'tstatic':      (pnl_exp_full_tstat,  pos_exp_full_tstat,  pos_exp_full_vol_tstat),
}

BASE_RESULTS = {name: run_base_pipeline(*args) for name, args in SIGNAL_DEFS.items()}
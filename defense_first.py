# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 14:21:54 2025

@author: thiago.onohara
"""

import pandas as pd
import numpy as np 
import sys
import os
import matplotlib.pyplot as plt 
sys.path.append('F:/Front/Moedas/Base/')
from FX import database
import math 

bolsas = {
      'ES1 Index': 'S&P',
      'NQ1 Index': 'NQ',    
  }

comdtys = {
           'C 1 Comdty': 'Corn', 
           'SB1 Comdty': 'Sugar', 
           'KC1 Comdty': 'Coffee', 
           'CC1 Comdty': 'Cocoa',
           'CT1 Comdty': 'Cotton', 
           'LB1 Comdty': 'Lumber',
           'XB1 Comdty': 'Gasoline',
           'CL1 Comdty': 'WTI', 
           'GC1 Comdty': 'Gold', 
           'HG1 Comdty': 'Copper', 
           'LA1 Comdty': 'Aluminum', 
           }
    
long_term_rates = {
    'TY1 Comdty':'10y'
    }

cash = {
        'USGG3M Index':'US_03m'
        }
   
dxy = {
       'DXY Curncy':'DXY',
       }

currencies = {
    "AUDUSDCR CMPN Curncy": "USDAUD",
    "USDCADCR CMPN Curncy": "USDCAD",
    "EURUSDCR CMPN Curncy": "USDEUR",
    "GBPUSDCR CMPN Curncy": "USDGBP",
    "USDCHFCR CMPN Curncy": "USDCHF",
    "USDJPYCR CMPN Curncy": "USDJPY",
    "USDNOKCR CMPN Curncy": "USDNOK",
    "NZDUSDCR CMPN Curncy": "USDNZD",
    "USDSEKCR CMPN Curncy": "USDSEK",
}

currencies_factor = {
    "AUDUSDCR CMPN Curncy": -1,
    "USDCADCR CMPN Curncy": 1,
    "EURUSDCR CMPN Curncy": -1,
    "GBPUSDCR CMPN Curncy": -1,
    "USDCHFCR CMPN Curncy": 1,
    "USDJPYCR CMPN Curncy": 1,
    "USDNOKCR CMPN Curncy": 1,
    "NZDUSDCR CMPN Curncy": -1,
    "USDSEKCR CMPN Curncy": 1,
}

curncy_df = database.getData(tickers=currencies.keys()).pivot_table(
    index="date", columns="security", values="PX_LAST")
curncy_df_px = curncy_df.pow(currencies_factor).rename(columns=currencies)
curncy_df_px['USDG10'] = curncy_df_px.product(axis=1)**(1/9)

curncy_df_px['Safe'] = 1/curncy_df_px[['USDJPY', 'USDEUR', 'USDCHF']].product(axis=1)**(1/3)

vix = {'UX1 Index':'VIX'}

usd = {'Safe':'Safe'}


assets = {**bolsas, **comdtys, **long_term_rates, **cash, **usd, **vix, **dxy}

def get_prices():
    prices_df = database.getData(tickers=assets.keys())
    return prices_df

#%%
prices_df = get_prices()

prices_df['name'] = prices_df['security'].map(assets)
prices_tseries = prices_df.pivot_table(index='date', columns='name', values='PX_LAST')
prices_tseries_full = pd.concat([prices_tseries, curncy_df_px[['USDG10']]], axis=1)
prices_tseries_gold = prices_tseries_full[['Gold']]
prices_tseries_tsy = prices_tseries_full[['10y']]
prices_tseries_dxy = prices_tseries_full[['DXY']]
prices_tseries_tbills = prices_tseries_full[['US_03m']]
prices_tseries_spx = prices_tseries_full[['S&P']]
prices_tseries_vix = prices_tseries_full[['VIX']]
prices_tseries_comdty = prices_tseries_full[['WTI', 'Copper', 'Gasoline']].mean(axis=1).to_frame('Comdty')

returns_full = prices_tseries_full.pct_change()
returns_full

returns_full_vol_adj = returns_full/returns_full.rolling(2*252, min_periods=252).std().shift(2)
returns_full_vol_adj

def generate_frame_mmt(df_px:pd.DataFrame):
    asset = df_px.columns[0]
    wdw_1m = 21
    wdw_3m = 63
    wdw_6m = 126
    wdw_12m = 252
    
    df_features = df_px.copy()
    df_features['mmt_01m'] = df_features[asset].pct_change(wdw_1m)/df_features[asset].pct_change(wdw_1m).rolling(252*2, min_periods=252).std()
    df_features['mmt_03m'] = df_features[asset].pct_change(wdw_3m)/df_features[asset].pct_change(wdw_3m).rolling(252*2, min_periods=252).std() 
    df_features['mmt_06m'] = df_features[asset].pct_change(wdw_6m)/df_features[asset].pct_change(wdw_6m).rolling(252*2, min_periods=252).std() 
    df_features['mmt_12m'] = df_features[asset].pct_change(wdw_12m)/df_features[asset].pct_change(wdw_12m).rolling(252*2, min_periods=252).std()
    
    mmt_wdw_ls = [
        'mmt_01m', 
        'mmt_03m',
        'mmt_06m',
        'mmt_12m']
    
    df_features_notna = df_features[mmt_wdw_ls].dropna()
    df_features_notna[asset] = df_features_notna[mmt_wdw_ls].mean(axis=1)
    
    return df_features_notna[[asset]]

prices_tseries_gold_mmt = generate_frame_mmt(prices_tseries_gold)
prices_tseries_tsy_mmt = generate_frame_mmt(prices_tseries_tsy)
prices_tseries_dxy_mmt = generate_frame_mmt(prices_tseries_dxy)
prices_tseries_comdty_mmt = generate_frame_mmt(prices_tseries_comdty)
prices_tseries_spx_mmt = generate_frame_mmt(prices_tseries_spx)
prices_tseries_vix_mmt = generate_frame_mmt(prices_tseries_vix)

features = pd.concat([
    prices_tseries_gold_mmt,
    prices_tseries_tsy_mmt,
    prices_tseries_dxy_mmt,
    prices_tseries_comdty_mmt
    ], axis=1).loc[:]

defensive_assets = ['Gold', '10y', 'DXY', 'Comdty']
SIZE_DISTRB = 1

port_allocation = {1:0.4*SIZE_DISTRB, 
                   2:0.3*SIZE_DISTRB, 
                   3:0.2*SIZE_DISTRB, 
                   4:0.1*SIZE_DISTRB,
                   5:0.0*SIZE_DISTRB}

features_signal = features.shift(2).round(4)

features_signal_alloc = features_signal[features_signal.gt(0)].rank(axis=1, ascending=False).map(lambda x: port_allocation.get(x))
features_signal_alloc['S&P'] = features_signal_alloc.apply(lambda row: 1-row.sum(), axis=1)
features_signal_alloc_ = features_signal_alloc.round(1)

returns_full_vol_adj_assets = returns_full_vol_adj.rename(columns={'WTI':'Comdty'})[features_signal_alloc.columns]

pnl_composition = (features_signal_alloc_.fillna(0)*returns_full_vol_adj_assets)
pnl_composition
#Custo como 0.02 vols
pnl_composition_custo = pnl_composition.fillna(0) - (features_signal_alloc_.fillna(0).diff().abs().mul(0.02))
pnl_composition_custo

features_signal_alloc_.plot(kind='area', subplots=True, alpha=0.35, figsize=(8, 4), sharey=True)


#%%
def plot_pnl_summary(pnl_df, strat=True, mensal=False, out_of_sample=None):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    fig = plt.figure(constrained_layout=True, figsize=(6, 11))
    fig.suptitle('Strategy Summary', fontsize=16, fontweight='bold')
    gs = GridSpec(4, 4, figure=fig)
    ax1 = fig.add_subplot(gs[:2, :])
    ax2 = fig.add_subplot(gs[2:3, :])
    ax3 = fig.add_subplot(gs[3:, :])
    PNL = pnl_df[pnl_df != 0].dropna(how='all').copy()
    
    if strat:
        PNL['Strat'] = PNL.sum(axis=1)
        dd_strat = (PNL[['Strat']].cumsum()-PNL[['Strat']].cumsum().cummax())
        PNL[['Strat']].cumsum().ffill().plot(ax=ax2, title="PnL-Strat")
        dd_strat.plot(ax=ax3, title="Drawdown")
    else:
        dd = (PNL.cumsum()-PNL.cumsum().cummax())
        dd.plot(ax=ax3, title="Drawdown")
        
    PNL.cumsum().ffill().drop('Strat', axis=1).plot(ax=ax1)
    ax1.title.set_text('PnL (Vol Adjusted)')
    ax2.title.set_text('PnL Strategy')
    ax3.title.set_text('Drawdown')
    ax1.title.set_fontweight('bold')
    ax2.title.set_fontweight('bold')
    ax3.title.set_fontweight('bold')
    
    if out_of_sample is not None:
        print('pnl out of sample:', out_of_sample)
        ax1.axvline(x=out_of_sample, lw=1, ls='dashed', color='red')
        ax2.axvline(x=out_of_sample, lw=1, label='oos_start' , ls='dashed', color='red')
        ax3.axvline(x=out_of_sample, lw=1, ls='dashed', color='red')
        
    ax1.grid(axis='both')
    ax2.grid(axis='both')
    ax3.grid(axis='both')

    if not mensal:
        SHARPE = PNL.apply(lambda serie: np.sqrt(252)*serie[serie != 0].dropna().mean()/serie[serie != 0].dropna().std())
    else:
        SHARPE = PNL.apply(lambda serie: np.sqrt((3*12))*serie[serie != 0].dropna().mean()/serie[serie != 0].dropna().std())
    print(SHARPE)
    
    ax1.legend(labels=[
        c+'={}'.format(round(s, 2))
        for c, s in zip(SHARPE.index, SHARPE.values)],
        ncol=2, fontsize=10, loc='upper left',
        title='Sharpe')
    
    ax2.legend(labels=[
        c+'={}'.format(round(s, 2))
        for c, s in zip(SHARPE[['Strat']].index, SHARPE[['Strat']].values)],
        ncol=2, fontsize=10, loc='upper left',
        title='Sharpe')
    
    return fig, SHARPE, PNL

#%%
sharpe_func = lambda series: np.sqrt(252)*series.mean()/series.std()

#plot_pnl_summary(pnl_composition)
plot_pnl_summary(pnl_composition_custo)
plot_pnl_summary(pnl_composition_custo.loc['2026':])

pnl_composition_ = pnl_composition.copy()
pnl_composition_['Strat'] = pnl_composition_.sum(axis=1)
pnl_composition_.apply(sharpe_func)

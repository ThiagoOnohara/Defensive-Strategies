# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 16:00:24 2026

@author: thiago.onohara
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import sys
sys.path.append('F:/Front/Moedas/Base/')
from FX import database

#%%
def plot_pnl(pnl_df, active_sharpe=True, vol_adj=False):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    
    fig = plt.figure(constrained_layout=True, figsize=(6,11))
    fig.suptitle('Strategy Summary', fontsize=16, fontweight='bold')
    gs = GridSpec(4, 4, figure=fig)
    ax1 = fig.add_subplot(gs[:2,:])
    ax2 = fig.add_subplot(gs[2:3,:])
    ax3 = fig.add_subplot(gs[3:,:])

    PNL = pnl_df.copy()#[pnl_df!=0]
    if vol_adj:
        PNL = PNL/PNL.std()
    PNL['Strat'] = PNL.mean(axis=1)
    print(PNL.cumsum().ffill())
    PNL.drop('Strat', axis=1).cumsum().ffill().plot(ax=ax1)
    dd_strat = (PNL[['Strat']].cumsum()-PNL[['Strat']].cumsum().cummax())
    PNL[['Strat']].cumsum().ffill().plot(ax=ax2, title="PnL-Strat")
    dd_strat.plot(ax=ax3, title="Drawdown")

    ax1.title.set_text('PnL (Vol Not Adjusted)')
    ax2.title.set_text('PnL Strategy')
    ax3.title.set_text('Drawdown')
    ax1.title.set_fontweight('bold')
    ax2.title.set_fontweight('bold')
    ax3.title.set_fontweight('bold')
    
    ax1.grid(axis='both')
    ax2.grid(axis='both')
    ax3.grid(axis='both')
    
    SHARPE = PNL.apply(lambda serie: np.sqrt(252)*serie[serie!=0].dropna().mean()/serie[serie!=0].dropna().std())
    
    if not active_sharpe: 
        SHARPE = PNL.apply(lambda serie: np.sqrt(252)*serie.dropna().mean()/serie.dropna().std())
        
    print(SHARPE)    
    return PNL

#%%    
currencies = {
    'USDBRLCR CMPN Curncy': 'BRL',
    'USDMXNCR CMPN Curncy': 'MXN',
    'USDCLPCR CMPN Curncy': 'CLP',
    'USDCOPCR CMPN Curncy': 'COP',
    'USDCNYCR CMPN Curncy': 'CNY',
    'USDINRCR CMPN Curncy': 'INR',
    'USDKRWCR CMPN Curncy': 'KRW',
    'USDSGDCR CMPN Curncy': 'SGD',
    'USDIDRCR CMPN Curncy': 'IDR',
    'USDTHBCR CMPN Curncy': 'THB',
    'USDTWDCR CMPN Curncy': 'TWD',
    'USDCZKCR CMPN Curncy': 'CZK',
    'USDHUFCR CMPN Curncy': 'HUF',
    'USDPLNCR CMPN Curncy': 'PLN',
    'USDZARCR CMPN Curncy': 'ZAR',
    'AUDUSDCR CMPN Curncy': 'AUD',
    'USDCADCR CMPN Curncy': 'CAD',
    'EURUSDCR CMPN Curncy': 'EUR',
    'GBPUSDCR CMPN Curncy': 'GBP',
    'USDCHFCR CMPN Curncy': 'CHF',
    'USDJPYCR CMPN Curncy': 'JPY',
    'USDNOKCR CMPN Curncy': 'NOK',
    'NZDUSDCR CMPN Curncy': 'NZD',
    'USDSEKCR CMPN Curncy': 'SEK'}

currencies_factor = {
    'USDBRLCR CMPN Curncy': -1,
    'USDMXNCR CMPN Curncy': -1,
    'USDCLPCR CMPN Curncy': -1,
    'USDCOPCR CMPN Curncy': -1,
    'USDCNYCR CMPN Curncy': -1,
    'USDINRCR CMPN Curncy': -1,
    'USDKRWCR CMPN Curncy': -1,
    'USDSGDCR CMPN Curncy': -1,
    'USDIDRCR CMPN Curncy': -1,
    'USDTHBCR CMPN Curncy': -1,
    'USDTWDCR CMPN Curncy': -1,
    'USDCZKCR CMPN Curncy': -1,
    'USDHUFCR CMPN Curncy': -1,
    'USDPLNCR CMPN Curncy': -1,
    'USDZARCR CMPN Curncy': -1,
    'AUDUSDCR CMPN Curncy': 1,
    'USDCADCR CMPN Curncy': -1,
    'EURUSDCR CMPN Curncy': 1,
    'GBPUSDCR CMPN Curncy': 1,
    'USDCHFCR CMPN Curncy': -1,
    'USDJPYCR CMPN Curncy': -1,
    'USDNOKCR CMPN Curncy': -1,
    'NZDUSDCR CMPN Curncy': 1,
    'USDSEKCR CMPN Curncy': -1}


bolsas = {
          'ES1 Index':'S&P',
          'BZ1 Index':'Ibov',
          'CF1 Index':'CAC 40',
          'SM1 Index':'Swiss',
          'GX1 Index':'DAX',
          'VG1 Index':'Stoxx',
          'XP1 Index':'SPI 200', 
          'Z 1 Index':'FTSE 100',
          'NK1 Index':'Nikkei',
          'NQ1 Index': 'NQ',
          'EEM Index': 'EEM'
          }


commodities = {
    'S 1 Comdty': 'Soybean',
    'W 1 Comdty': 'Wheat',
    'C 1 Comdty': 'Corn',
    'SB1 Comdty': 'Sugar',
    'KC1 Comdty': 'Coffee',
    'LH1 Comdty': 'Lean Hogs',
    'LC1 Comdty': 'Live Cattle',
    'CC1 Comdty': 'Cocoa',
    'CT1 Comdty': 'Cotton',
    'FC1 Comdty': 'Cattle Feeder',
    'KW1 Comdty': 'Winter Wheat',
    'O 1 Comdty': 'Oats',
    'HO1 Comdty': 'Heating Oil',
    'NG1 Comdty': 'Natural Gas',
    'XB1 Comdty': 'Gasoline',
    'CO1 Comdty': 'Brent',
    'CL1 Comdty': 'WTI',
    'GC1 Comdty': 'Gold',
    'SI1 Comdty': 'Silver',
    'HG1 Comdty': 'Copper',
    'PL1 Comdty': 'Platinum',
    'LN1 Comdty': 'Nickel'}

rates = {
    'TY1 Comdty': 'Bond_USD',
    'G 1 Comdty': 'Bond_GBP',
    'JB1 Comdty': 'Bond_JPY',
    'CN1 Comdty': 'Bond_CAD',
    'RX1 Comdty': 'Bond_GER',
    'OAT1 Comdty':'Bond_FRA',
    'IK1 Comdty': 'Bond_ITL'}

px_tickers = {**currencies, **bolsas, **commodities, **rates}
agro_futures = ['Soybean', 'Wheat', 'Corn', 'Sugar',
                'Coffee', 'Lean Hogs', 'Live Cattle', 'Cotton', 'Cattle Feeder', 'Cocoa', 'Oats', 'Winter Wheat']
energy_futures = ['Natural Gas', 'Gasoline', 'Brent', 'WTI', 'Heating Oil']
metals_futures = ['Gold', 'Silver', 'Copper', 'Palladium', 'Platinum']

px_data = database.getData(tickers=px_tickers.keys())

curncy_px = px_data[px_data['security'].isin(currencies.keys())].pivot_table(
    index='date', columns='security', values='PX_LAST')
for column, factor in currencies_factor.items():
    curncy_px[column] = curncy_px[column].copy()**(factor)
curncy_px = curncy_px.rename(columns=currencies).ffill()

bolsas_px = px_data[px_data['security'].isin(bolsas.keys())].pivot_table(
    index='date', columns='security', values='PX_LAST').rename(columns=bolsas).ffill()    

comdty_px = px_data[px_data['security'].isin(commodities.keys())].pivot_table(
    index='date', columns='security', values='PX_LAST').rename(columns=commodities).drop(agro_futures, axis=1).ffill()

rates_px = px_data[px_data['security'].isin(rates.keys())].pivot_table(
    index='date', columns='security', values='PX_LAST').rename(columns=rates).ffill()  

#%% Price and Baskets

total_px = pd.concat([curncy_px, bolsas_px, comdty_px, rates_px], axis=1)
total_rt = np.log(total_px.ffill()/total_px.shift(1).ffill()) 

total_rt_vol_adj = total_rt/total_rt.rolling(2*252, min_periods=252).quantile(0.025).abs().shift(2)
            
#Generate Tracking Portfolios
eq_vol = total_rt['S&P'].rolling(2*252, min_periods=252).std().shift(2)
bd_vol = total_rt['Bond_USD'].rolling(2*252, min_periods=252).std().shift(2)
eq_bd_vol_ratio = eq_vol/bd_vol
bd_vol_as_eq = total_rt['Bond_USD']*eq_bd_vol_ratio
equity_only = total_rt['S&P'].to_frame('100/0')
#portf_60_40 = (total_rt['S&P']*0.6 + bd_vol_as_eq*0.4).to_frame('60/40')
portf_50_50 = (total_rt['S&P']*0.5 + bd_vol_as_eq*0.5).to_frame('50/50')

portfolios_tracking = pd.concat(
    [
     equity_only,
     #portf_60_40,
     portf_50_50,
     ], axis=1).dropna(axis=0)
portfolios_tracking.cumsum().plot()
portfolios_tracking.columns.name = 'portfolios'

total_rt_def = pd.concat(
    [
     total_rt_vol_adj,
     #trend_signal[['TR_Bond', 'TR_Eqty', 'TR_Energy', 'TR_Agro', 'TR_Metals']],
     #rev_signal[['Value_Bond', 'Value_Eqty', 'Value_Energy', 'Value_Agro', 'Value_Metals']]
     ], axis=1)
total_rt_def

sharpe_func = lambda s: np.sqrt(252)*s.mean()/s.std()
sortino_func = lambda s: np.sqrt(252)*s.mean()/s[s<0].std()

metric_func = sortino_func

hedging_corr_hist = pd.concat(
    map(
        lambda df: (metric_func(df.add(df['S&P'].reindex(df.index), axis=0))-
                    metric_func(df['S&P'].reindex(df.index))).to_frame(df.index[-1]).T, 
    total_rt_vol_adj.rolling(20, min_periods=20)))

hedging_corr_hist_valid = hedging_corr_hist.shift(2).dropna(how='all')
hedging_corr_hist_valid

hedges_ponta = 5

hedging_composition_hist = hedging_corr_hist_valid.drop('S&P', axis=1)*np.nan
hedging_composition_hist[hedging_corr_hist_valid.rank(axis=1, ascending=False).le(hedges_ponta)] = 1
hedging_composition_hist[hedging_corr_hist_valid.rank(axis=1, ascending=True).le(hedges_ponta)] = -1
hedging_composition_hist
hedging_composition_hist.tail(3).dropna(how='all', axis=1).T

hedge_rets_hist = hedging_composition_hist*total_rt_vol_adj
hedge_rets_hist.index.name = 'date'
hedge_rets_hist.columns.name = 'security'
dinamic_hedge_hist = hedge_rets_hist.stack().groupby('date').mean().to_frame('hedge_hist')
plot_pnl(dinamic_hedge_hist)

strat_hist = pd.concat([equity_only, 
                        (dinamic_hedge_hist['hedge_hist']
                         *equity_only['100/0'].rolling(252, min_periods=126).std().shift(2)
                         /dinamic_hedge_hist['hedge_hist'].rolling(252, min_periods=126).std().shift(2)
                         )], axis=1).dropna()

plot_pnl(strat_hist)

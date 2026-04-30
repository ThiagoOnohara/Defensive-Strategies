# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

import sys
sys.path.append('F:/Front/Moedas/Working/Sazonalidade/')
from roll_saz_strat_ import (
    BASE_RESULTS, apply_beta_hedging, plot_pnl,
    build_forward_position_projection, FORWARD_POSITION_DAYS,
    groups, eqw_groups, rp_groups,
    g10, ems, ems_asia,
    equity_cols, dms_rates, agro_futures, energy_futures, metals_futures,
    cross_g10_series, cross_ems_series, cross_ems_asia_series,
    pnl_exp_full_hit, pnl_exp_full_tr_hit, pnl_exp_full_tstat,
)

st.set_page_config(page_title='Rolling Seasonality', layout='wide')

SIGNAL_LABELS = {
    'hitratio':     'Hit Ratio (h5)',
    'trd_hitratio': 'Trade Hit Ratio (h3)',
    'tstatic':      'T-Stat (h5)',
}
SIGNAL_KEYS = list(BASE_RESULTS.keys())

# 3x3 panel layout
PANEL_LAYOUT = [
    ['Equity',   'Rates',      'Energy'],
    ['Metals',   'Agro',       'FX G10'],
    ['FX EMs',   'FX EMsAsia', 'Beta'],
]

# (source, group_col, asset_list) — used to filter fin_pos_asset columns
PANEL_ASSET_MAP = {
    'Equity':     list(equity_cols),
    'Rates':      dms_rates,
    'Energy':     energy_futures,
    'Metals':     metals_futures,
    'Agro':       agro_futures,
    'FX G10':     g10,
    'FX EMs':     ems,
    'FX EMsAsia': ems_asia,
    'Beta':       None,  # from beta_hedge_notional_per_bm
}

# ============================================================
# HELPERS
# ============================================================

def _snapshot(df, date_ref):
    return df.loc[:str(date_ref)].dropna(how='all').tail(1)


def small_pnl_fig(pnl_df, title='', figsize=(4.5, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    pnl_df.cumsum().ffill().plot(ax=ax, linewidth=1)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def summary_stats(pnl_df):
    ann = 252
    cum = pnl_df.cumsum()
    return pd.DataFrame({
        'Sharpe':  (pnl_df.mean() / pnl_df.std() * np.sqrt(ann)).round(2),
        'Ann PnL': (pnl_df.mean() * ann).round(1),
        'Ann Vol': (pnl_df.std()  * np.sqrt(ann)).round(1),
        'Max DD':  (cum - cum.cummax()).min().round(1),
    })


def _df_gradient(df, fmt='{:.1f}'):
    return df.style.background_gradient(cmap='RdYlGn', axis=None).format(fmt)


def _format_forward_table(df, row_name, sort_by_first_col=True):
    """Convert a date x asset/benchmark matrix into the dashboard table view."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.index = pd.to_datetime(out.index).normalize()
    out = out.T
    out.columns = [c.strftime('%Y-%m-%d') for c in out.columns]
    out = out.replace([np.inf, -np.inf], np.nan)

    # Remove assets/benchmarks that have no position in the whole forward window.
    active = out.fillna(0).abs().sum(axis=1) != 0
    out = out.loc[active]
    if out.empty:
        return out
    out = out.fillna(0)

    if sort_by_first_col and len(out.columns) > 0:
        first_col = out.columns[0]
        out = out.assign(_sort_key=out[first_col].fillna(0)).sort_values('_sort_key').drop(columns='_sort_key')

    out.index.name = row_name
    return out


# ============================================================
# 3x3 POSITION PANEL
# ============================================================

def render_position_panel(base, hedge_result, date_ref, div_factor,
                          forward_days=FORWARD_POSITION_DAYS):
    projection = build_forward_position_projection(
        base, date_ref, days=forward_days)
    fin_asset = projection['fin_pos_asset']

    # Same beta hedge notional convention as apply_beta_hedging():
    # beta_hedge_notional_per_bm = -portfolio_beta / div_factor.
    beta_forward = -projection['portfolio_beta'] / div_factor
    beta_forward = beta_forward.rename(
        columns=lambda c: c.replace('beta_', '') if isinstance(c, str) else c)

    for row_panels in PANEL_LAYOUT:
        cols = st.columns(3)
        for col, panel in zip(cols, row_panels):
            with col:
                st.markdown(f'**{panel}**')

                if panel == 'Beta':
                    df_show = _format_forward_table(beta_forward, 'benchmark')
                    if df_show.empty:
                        st.caption('no data')
                        continue
                    st.dataframe(_df_gradient(df_show), use_container_width=True)

                else:
                    asset_list = PANEL_ASSET_MAP[panel]
                    valid = [a for a in asset_list if a in fin_asset.columns]
                    if not valid:
                        st.caption('no data')
                        continue
                    df_show = _format_forward_table(fin_asset[valid], 'asset')
                    if df_show.empty:
                        st.caption('no data')
                        continue
                    st.dataframe(_df_gradient(df_show), use_container_width=True)


# ============================================================
# SHAP BAR CHART
# ============================================================

def render_shap_chart(base):
    shap    = base['shap_sharpe']
    sh_full = base['sharpe_full']

    fig, ax = plt.subplots(figsize=(4.5, 3))
    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in shap.values]
    ax.barh(shap.index, shap.values, color=colors, edgecolor='none')
    ax.axvline(0, color='white', linewidth=0.8, alpha=0.5)
    ax.set_title(
        f'SHAP Sharpe by group  (Full = {sh_full:.2f})',
        fontsize=8, fontweight='bold')
    ax.set_xlabel('Sharpe contribution', fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title('Controls')

min_date = min(BASE_RESULTS[k]['pnl_raw_groups'].index.min() for k in SIGNAL_KEYS)
last_day = datetime.today().replace(day=31, month=12)

date_show = st.sidebar.date_input(
    'Reference date', value='today', min_value=min_date, max_value=last_day)

div_factor = st.sidebar.select_slider(
    'Beta hedge div factor',
    options=[1.0, 1.5, 2.0, 2.5, 3.0], value=2.0,
    help='1 = full beta notional, 3 = one-third.')

show_since = st.sidebar.date_input(
    'OOS start date', value=pd.Timestamp('2024-01-01'))

# ============================================================
# APPLY BETA HEDGING
# ============================================================

hedged = {k: apply_beta_hedging(BASE_RESULTS[k], div_factor) for k in SIGNAL_KEYS}

# ============================================================
# TITLE
# ============================================================

st.title('Rolling Seasonality Dashboard')
st.caption(f'Reference date: {date_show}  |  Beta div factor: {div_factor}')
st.divider()

# ============================================================
# SECTION 1 — Signal comparison
# ============================================================

st.subheader('Signal Comparison')
tab_full, tab_oos = st.tabs(['Full Sample', f'OOS from {show_since}'])

for tab_obj, sl in [(tab_full, slice(None)), (tab_oos, slice(str(show_since), None))]:
    with tab_obj:
        cols = st.columns(3)
        for col, key in zip(cols, SIGNAL_KEYS):
            base_  = BASE_RESULTS[key]
            hedge_ = hedged[key]
            with col:
                st.markdown(f'### {SIGNAL_LABELS[key]}')

                fig_cls = small_pnl_fig(base_['pnl_dir_var_grp'].loc[sl], 'Class PnL (VaR adj)')
                st.pyplot(fig_cls, use_container_width=True); plt.close(fig_cls)

                fig_h = small_pnl_fig(
                    hedge_['pnl_scenarios'][
                        ['no_hedged', 'hedged_sp500', 'hedged_treasury', 'hedged_full']
                    ].loc[sl],
                    f'Beta-Hedged PnL (÷{div_factor})')
                st.pyplot(fig_h, use_container_width=True); plt.close(fig_h)

                st.dataframe(summary_stats(hedge_['pnl_scenarios'].loc[sl]).T,
                             use_container_width=True)

st.divider()

# ============================================================
# SECTION 2 — Positions (3x3 panel, per signal in expanders)
# ============================================================

st.subheader(f'Positions & Beta Hedge — {date_show} + next {FORWARD_POSITION_DAYS} business days')

for key in SIGNAL_KEYS:
    with st.expander(f'📋 {SIGNAL_LABELS[key]}', expanded=(key == 'hitratio')):
        render_position_panel(BASE_RESULTS[key], hedged[key], date_show, div_factor)

st.divider()

# ============================================================
# SECTION 3 — SHAP Sharpe
# ============================================================

st.subheader('SHAP Sharpe — Group Contribution')
shap_cols = st.columns(3)

for col, key in zip(shap_cols, SIGNAL_KEYS):
    with col:
        st.markdown(f'**{SIGNAL_LABELS[key]}**')

        fig_shap = render_shap_chart(BASE_RESULTS[key])
        st.pyplot(fig_shap, use_container_width=True); plt.close(fig_shap)

        shap_df = (BASE_RESULTS[key]['shap_sharpe']
                   .to_frame('SHAP Sharpe')
                   .sort_values('SHAP Sharpe', ascending=False))
        st.dataframe(
            shap_df.style.background_gradient(cmap='RdYlGn', axis=None).format('{:.3f}'),
            use_container_width=True)

st.divider()

# ============================================================
# SECTION 4 — Full strategy PnL (tabs per signal)
# ============================================================

st.subheader('Full Strategy PnL')
strat_tabs = st.tabs([SIGNAL_LABELS[k] for k in SIGNAL_KEYS])

for tab, key in zip(strat_tabs, SIGNAL_KEYS):
    base_  = BASE_RESULTS[key]
    hedge_ = hedged[key]

    with tab:
        c1, c2 = st.columns(2)
        pnl_grp = base_['pnl_grouped'].pivot_table(index='date', columns='asset', values='pnl')

        c1.markdown('**Class PnL (group-level VaR)**')
        c1.pyplot(plot_pnl(pnl_grp[groups], return_fig=True), use_container_width=True)
        plt.close()

        c2.markdown(f'**Hedged Scenarios (÷{div_factor})**')
        c2.pyplot(plot_pnl(hedge_['pnl_scenarios'], return_fig=True), use_container_width=True)
        plt.close()

        # Beta exposure over time
        st.markdown(f'**Normalised beta exposure (÷{div_factor})**')
        fig_beta, ax = plt.subplots(figsize=(10, 3))
        hedge_['portfolio_beta_exp_'].plot(ax=ax, linewidth=1)
        ax.axhline(0, color='white', linewidth=0.5, alpha=0.4)
        ax.legend(fontsize=7, ncol=4)
        ax.grid(alpha=0.3)
        fig_beta.tight_layout()
        st.pyplot(fig_beta, use_container_width=True); plt.close(fig_beta)

        # Beta hedge full notional over time
        st.markdown('**Beta hedge full notional over time**')
        fig_bhn, ax2 = plt.subplots(figsize=(10, 2.5))
        hedge_['beta_hedge_full'].plot(ax=ax2, color='#f39c12', linewidth=1)
        ax2.axhline(0, color='white', linewidth=0.5, alpha=0.4)
        ax2.set_title('hedge_full notional', fontsize=8)
        ax2.grid(alpha=0.3)
        fig_bhn.tight_layout()
        st.pyplot(fig_bhn, use_container_width=True); plt.close(fig_bhn)

        st.dataframe(hedge_['pnl_scenarios'].dropna(how='all'), use_container_width=True)
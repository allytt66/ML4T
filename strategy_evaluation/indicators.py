# output indicators as functions operate on DataFrames.
# main method generate the chart that will illustrate indicators in the report.
# Use the time period January 1, 2008, to December 31, 2009.

# for each indicator, create a single, compelling chart with proper title, legend, and axis labels that illustrate the indicator
# research different size and parameters, Bolling Bands, MA, RSI, CCI, MACD,


import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data


def bbp(sd, ed, symbol, lookback, df_price_full, plot=False):
    rolling_mean = df_price_full.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = df_price_full.rolling(window=lookback, min_periods=lookback).std()
    rolling_mean = rolling_mean.loc[sd:][symbol]
    rolling_std = rolling_std.loc[sd:][symbol]
    top_band = rolling_mean + (2 * rolling_std)
    bottom_band = rolling_mean - (2 * rolling_std)
    bbp_series = (df_price_full[sd:][symbol] - bottom_band) / (top_band - bottom_band)

    bbp_df = (
        pd.DataFrame(bbp_series)
        .rename(columns={symbol: "bbp"})
        .set_index(df_price_full[sd:].index)
    )
    bbp_df = bbp_df[sd:]
    return bbp_df


def price_sma(sd, ed, symbol, lookback, df_price_full, plot=False):
    rolling_mean = df_price_full.rolling(window=lookback, min_periods=lookback).mean()
    rolling_mean = rolling_mean.loc[sd:][symbol]
    price_sma_series = df_price_full[sd:][symbol] / rolling_mean

    sma_df = (
        pd.DataFrame(price_sma_series)
        .rename(columns={symbol: "sma"})
        .set_index(df_price_full[sd:].index)
    )
    sma_df = sma_df[sd:]


    return sma_df


def rsi(sd, ed, symbol, lookback, df_price_full, plot=False):
    moves = df_price_full.diff()[symbol]
    up = []
    down = []

    for i in range(len(moves)):
        if moves[i] < 0:
            up.append(0)
            down.append(moves[i])
        else:
            up.append(moves[i])
            down.append(0)
    up_series = pd.Series(up)
    down_series = pd.Series(down).abs()

    up_sma = up_series.rolling(window=lookback, min_periods=lookback).mean()
    down_sma = down_series.rolling(window=lookback, min_periods=lookback).mean()

    rs = up_sma / down_sma
    rsi_series = 100 - (100 / (1 + rs))
    rsi_df = (
        pd.DataFrame(rsi_series)
        .rename(columns={0: "rsi"})
        .set_index(df_price_full.index)
    )
    rsi_df = rsi_df[sd:]

    return rsi_df


def stochastic_osi(sd, ed, symbol, lookback, df_price_full, plot=False):
    extend_sd = sd - dt.timedelta(100)
    df_high = get_data(
        symbols=[symbol], dates=pd.date_range(extend_sd, ed), addSPY=True, colname="High"
    )[[symbol]]
    df_low = get_data(
        symbols=[symbol], dates=pd.date_range(extend_sd, ed), addSPY=True, colname="Low"
    )[[symbol]]
    df_close = get_data(
        symbols=[symbol],
        dates=pd.date_range(extend_sd, ed),
        addSPY=True,
        colname="Close",
    )[[symbol]]

    high_roll = df_high[symbol].rolling(lookback).max()
    low_roll = df_low[symbol].rolling(lookback).min()

    num = df_close[symbol] - low_roll
    den = high_roll - low_roll
    per_k = (num / den) * 100

    per_k_df = pd.DataFrame(per_k).rename(columns={symbol: "osi"}).set_index(df_high.index)
    per_k_df = per_k_df[sd:]

    return per_k_df


def chaikin(sd, ed, symbol, lookback, df_price_full, plot=False):
    extend_sd = sd - dt.timedelta(100)
    df_high = get_data(
        symbols=[symbol], dates=pd.date_range(extend_sd, ed), addSPY=True, colname="High"
    )[[symbol]]
    df_low = get_data(
        symbols=[symbol], dates=pd.date_range(extend_sd, ed), addSPY=True, colname="Low"
    )[[symbol]]
    df_close = get_data(
        symbols=[symbol],
        dates=pd.date_range(extend_sd, ed),
        addSPY=True,
        colname="Close",
    )[[symbol]]
    df_vol = get_data(
        symbols=[symbol],
        dates=pd.date_range(extend_sd, ed),
        addSPY=True,
        colname="Volume",
    )[[symbol]]

    clv = (
        df_vol[symbol]
        * (2 * df_close[symbol] - df_high[symbol] - df_low[symbol])
        / (df_high[symbol] - df_low[symbol])
    )

    chaikin_series = clv.rolling(lookback).sum() / df_vol[symbol].rolling(lookback).sum()
    chaikin_df = pd.DataFrame(chaikin_series).rename(columns={symbol: "chaikin"}).set_index(df_high.index)[sd:]

    return chaikin_df


def author():
    return "yzhang3946"

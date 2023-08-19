import numpy as np

import pandas as pd
from util import get_data, plot_data


def compute_portvals(
    orders_file="./orders/orders-09.csv",
    start_val=1000000,
    commission=0,
    impact=0,
):
    orders = pd.read_csv(orders_file)
    start, end, period = get_dates(orders)

    symbols = orders.Symbol.unique()
    df_price = get_prices(symbols, period)
    df_price["Cash"] = np.ones(df_price.shape[0])

    df_trade = pd.DataFrame().reindex_like(df_price).fillna(0)

    for i in orders.index:
        date = orders.iloc[i]["Date"]
        symbol = orders.iloc[i]["Symbol"]
        order = orders.iloc[i]["Order"]
        shares = orders.iloc[i]["Shares"]
        price = df_price.loc[date, symbol]

        if order == "SELL":
            shares = -shares
            cash_trade = -(price * (1 - impact) * shares) - commission
        else:
            cash_trade = -(price * (1 + impact) * shares) - commission

        df_trade.loc[date, symbol] += shares
        df_trade.loc[date, "Cash"] += cash_trade

    df_holdings = pd.DataFrame().reindex_like(df_price).fillna(0)

    for i in range(df_holdings.shape[0]):
        if i == 0:
            df_holdings.iloc[i] = df_trade.iloc[i]
            df_holdings.iloc[i]["Cash"] += start_val

        else:
            df_holdings.iloc[i] = df_holdings.iloc[i - 1] + df_trade.iloc[i]

    df_value = df_price * df_holdings
    portvals = df_value.sum(axis=1)

    return portvals


def get_dates(orders):
    start = orders.Date.iloc[0]
    end = orders.Date.iloc[-1]
    period = pd.date_range(start, end)
    return start, end, period


def get_prices(symbols, period):
    return get_data(symbols, period)


def test_code():
    of = "./orders/orders-01.csv"
    sv = 1000000

    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    print(portvals)


if __name__ == "__main__":
    test_code()


def author():
    return "yzhang3946"

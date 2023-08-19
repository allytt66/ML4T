## implement testPolicy(), which returns a trades data frame
## allow position 1000 share long, short, 0, up to 2000
## benchmarket, starting with 100,000 cash, invest in 1000 share of JPM, and hold
## create a set of trades represent best strategy could possibly do during the in-sample period. have an upper bound on performance.
## can see future, only constrained by portfolio size and order limits.
## does not use indicators developed.

## output a chart that reports, benchmark, normalized to 1.0 -> purple, value of theoretically optimal port -> red.

## cumulative return
## stdev of daily return
## mean of daily return


## output single column df, indexed by date, value represent trades for each trading day (+2000, +1000, -1000, -2000, 0 )

# testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000)


from util import get_data
from marketsimcode import compute_portvals
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt


def tos_trade(symbol, sd, ed, sv):
    df_price = get_data([symbol], pd.date_range(sd, ed))
    df_price = df_price[[symbol]]

    df_trade = pd.DataFrame().reindex_like(df_price).fillna(0)
    dates = df_trade.index

    current_position = 0

    for i in range(len(dates) - 1):
        if df_price.loc[dates[i + 1]].loc[symbol] > df_price.loc[dates[i]].loc[symbol]:
            action = 1000 - current_position
        else:
            action = -1000 - current_position
        df_trade.loc[dates[i]].loc[symbol] = action
        current_position += action

    return df_trade, df_price


def benchmark_trade(symbol, sd, ed, sv):
    df_price = get_data([symbol], pd.date_range(sd, ed))
    df_price = df_price[[symbol]]
    df_trade = pd.DataFrame().reindex_like(df_price).fillna(0)
    df_trade.iloc[0] = 1000

    return df_trade


def get_stats(portvals):
    cr = portvals[-1] / portvals[0] - 1
    dr = (portvals / portvals.shift(1) - 1).iloc[1:]
    sddr = dr.std()
    adr = dr.mean()
    return cr, sddr, adr


def testPolicy(
    symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000
):
    df_trade_tos, df_price = tos_trade(symbol, sd, ed, sv)
    df_trade_bm = benchmark_trade(symbol, sd, ed, sv)
    portvals_tos = compute_portvals(df_trade_tos, df_price)
    portvals_bm = compute_portvals(df_trade_bm, df_price)

    portvals_tos = portvals_tos / portvals_tos[0]
    portvals_bm = portvals_bm / portvals_bm[0]

    fig, ax = plt.subplots()
    ax.plot(portvals_tos, color="red", label="tos")
    ax.plot(portvals_bm, color="purple", label="benchmark")
    ax.set_title("Theoretically Optimal Strategy")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True)
    fig.set_figwidth(14)
    fig.savefig("tos.png")

    cr_tos, sddr_tos, adr_tos = get_stats(portvals_tos)
    cr_bm, sddr_bm, adr_bm = get_stats(portvals_bm)

    print("TOS")
    print("cumulative return: " + str(round(cr_tos, 6)))
    print("st dev daily return: " + str(round(sddr_tos, 6)))
    print("average daily return: " + str(round(adr_tos, 6)))

    print("Benchmark")
    print("cumulative return: " + str(round(cr_bm, 6)))
    print("st dev daily return: " + str(round(sddr_bm, 6)))
    print("average daily return: " + str(round(adr_bm, 6)))


def author():
    return "yzhang3946"


if __name__ == "__main__":
    testPolicy(
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=100000,
    )

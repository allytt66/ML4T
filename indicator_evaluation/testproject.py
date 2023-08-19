## contain entry point
# if "__name__" =="_main_"
# call the testPolicy function in TOS, as well as indicators and marketsim as need, to generate plots and stats

from TheoreticallyOptimalStrategy import testPolicy
from indicators import bbp, price_sma, rsi, stochastic_osi, chaikin
from util import get_data
import datetime as dt
import pandas as pd


if __name__ == "__main__":
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000

    testPolicy(
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=100000,
    )

    history = dt.timedelta(52)
    full_sd = sd - history
    df_price_full = get_data([symbol], pd.date_range(full_sd, ed))
    df_price_full = df_price_full[["JPM"]]

    bbp_series = bbp(sd, ed, 14, df_price_full)
    price_sma_series = price_sma(sd, ed, 14, df_price_full)
    rsi_df = rsi(sd, ed, 14, df_price_full)
    stochastic_per_k = stochastic_osi(sd, ed, 14, df_price_full)
    chaikin_indicator = chaikin(sd, ed, 14, df_price_full)


def author():
    return "yzhang3946"

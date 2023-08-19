import numpy as np

import pandas as pd
from util import get_data


def compute_portvals(
    df_trade,
    df_price,
    start_val=100000,
):
    df_price["Cash"] = np.ones(df_price.shape[0])
    df_trade["Cash"] = np.ones(df_price.shape[0])

    df_trade["Cash"] = -df_trade["JPM"] * df_price["JPM"]
    df_trade["Cash"] = df_trade["Cash"].cumsum() + start_val

    df_trade["JPM"] = df_trade["JPM"].cumsum()

    df_holding = df_price * df_trade
    portvals = df_holding.sum(axis=1)

    return portvals


def author():
    return "yzhang3946"

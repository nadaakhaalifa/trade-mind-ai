import pandas as pd


def load_btc_prices():
    df = pd.read_csv("data/btc.csv")

    # remove flat prices (no movement)
    df = df[df["Close"].diff() != 0]

    return df["Close"].tolist()
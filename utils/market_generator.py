import datetime
import numpy as np
import pandas_datareader as pdr
from esig import tosig
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler

class MarketGenerator:
    def __init__(self, ticker, start=datetime.date(2000, 1, 1), end=datetime.date(2019, 1, 1), freq="M"):

        self.ticker = ticker
        self.start = start
        self.end = end
        self.freq = freq

        self._load_data()

    def _load_data(self):
        try:
            self.data = pdr.get_data_yahoo(self.ticker, self.start, self.end)["Close"]
        except:
            raise RuntimeError(f"Could not download data for {self.ticker} from {self.start} to {self.end}.")

        self.windows = []
        for _, window in self.data.resample(self.freq):
            values = window.values# / window.values[0]
            path = leadlag(values)

            self.windows.append(path)

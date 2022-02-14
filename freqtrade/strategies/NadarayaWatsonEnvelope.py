# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
import math

import math
import logging

from datetime import datetime, timedelta, timezone
from timeit import default_timer as timer
from datetime import timedelta

def funcNadarayaWatsonEnvelope(dtloc, source = 'close', bandwidth = 8, window = 500, mult = 3):
    """
    // This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/
    // Nadaraya-Watson Envelope [LUX]
      https://www.tradingview.com/script/Iko0E2kL-Nadaraya-Watson-Envelope-LUX/
     :return: up and down   
     translated for freqtrade: viksal1982  viktors.s@gmail.com  
    """ 
    dtNWE = dtloc.copy()
    dtNWE['nwe_up'] = np.nan
    dtNWE['nwe_down'] =  np.nan
    wn = np.zeros((window, window))
    for i in range(window):
        for j in range(window):
                wn[i,j] = math.exp(-(math.pow(i-j,2)/(bandwidth*bandwidth*2)))
    sumSCW = wn.sum(axis = 1)
    def calc_nwa(dfr, init=0):
        global calc_src_value
        if init == 1:
            calc_src_value = list()
            return
        calc_src_value.append(dfr[source])
        mae = 0.0
        y2_val = 0.0
        y2_val_up = np.nan
        y2_val_down = np.nan
        if len(calc_src_value) > window:
            calc_src_value.pop(0)
        if len(calc_src_value) >= window:
            src = np.array(calc_src_value)
            sumSC = src * wn
            sumSCS = sumSC.sum(axis = 1)
            y2 = sumSCS / sumSCW
            sum_e = np.absolute(src - y2)
            mae = sum_e.sum()/window*mult 
            y2_val = y2[-1]
            y2_val_up = y2_val + mae
            y2_val_down = y2_val - mae
        return y2_val_up,y2_val_down
    calc_nwa(None, init=1)
    dtNWE[['nwe_up','nwe_down']] = dtNWE.apply(calc_nwa, axis = 1, result_type='expand')
    return dtNWE[['nwe_up','nwe_down']]


class NadarayaWatsonEnvelope(IStrategy):
   
    INTERFACE_VERSION = 2
    minimal_roi = {
        "0": 0.015
    }
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False

    window_buy = IntParameter(60, 1000, default=500, space='buy', optimize=True)
    bandwidth_buy = IntParameter(2, 15, default=8, space='buy', optimize=True)
    mult_buy = DecimalParameter(0.5, 20.0, default=3, space='buy', optimize=True)


    # Optimal timeframe for the strategy.
    timeframe = '5m'
    custom_info = {}
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    
    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
            'nwe_up': {'color': 'red'},
            'nwe_down': {'color': 'blue'}
        },
        'subplots': {
          
        }
    }
    def informative_pairs(self):

        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
       dataframe[['nwe_up','nwe_down']] = funcNadarayaWatsonEnvelope(dataframe, source = 'close', bandwidth = self.bandwidth_buy.value, window = self.window_buy.value, mult = self.mult_buy.value)
       return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['close'], dataframe['nwe_down'])) & 
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['close'], dataframe['nwe_up']    )) & 
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe

 
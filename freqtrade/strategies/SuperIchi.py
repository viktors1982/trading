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
from freqtrade.persistence import Trade, PairLocks
import math

import math
import logging

from datetime import datetime, timedelta, timezone
from timeit import default_timer as timer
from datetime import timedelta







def funcSuperIchi(dtloc, source = 'close', length = 14, mult = 2):
    """
    // This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/
    // SuperIchi [LUX]
      https://www.tradingview.com/script/vDGd9X9y-SuperIchi-LUX/
     :return: avg
     translated for freqtrade: viksal1982  viktors.s@gmail.com  
    """ 


    dtfT = dtloc.copy()
    dtfT['TR'] = ta.TRANGE(dtfT)
    dtfT['ATR'] = dtfT['TR'].ewm(alpha=1 / length).mean()  * mult
    dtfT['up'] =  ((dtfT['high'] + dtfT['low']) / 2 ) + dtfT['ATR']
    dtfT['dn'] =  ((dtfT['high'] + dtfT['low']) / 2 ) - dtfT['ATR']

    def calcFt(dfr, init=0):
        global calc_source
        global calc_upper
        global calc_lower
        global calc_os
        global calc_max
        global calc_min
        global calc_spt
        global calc_prev_spt
        if init == 1:
            calc_source  = 0.0
            calc_upper   = 0.0
            calc_lower   = 0.0
            calc_os      = 0
            calc_max     = 0.0
            calc_min     = 9999999999999999.0
            calc_spt     = 0.0
            calc_prev_spt = 0.0
            return
        if calc_source < calc_upper:
            if dfr['up'] < calc_upper:
                calc_upper = dfr['up'] 
        else:
            calc_upper = dfr['up']
        if calc_source > calc_lower:
            if dfr['dn'] > calc_lower:
                calc_lower =  dfr['dn']
        else:
            calc_lower = dfr['dn'] 
        
        if dfr[source] > calc_upper:
           calc_os = 1
        elif dfr[source] < calc_lower: 
           calc_os = 0
        
        calc_prev_spt = calc_spt
        if calc_os == 1:
            calc_spt = calc_lower
        else:
            calc_spt = calc_upper
        
        is_crossed = False
        

        if dfr[source] > calc_prev_spt and dfr[source] < calc_spt:
            is_crossed = True
        if dfr[source] < calc_prev_spt and dfr[source] > calc_spt:
            is_crossed = True

        if is_crossed == True:
            if dfr[source] > calc_max:
                calc_max = dfr[source]
        elif calc_os == 1:
            if dfr[source] > calc_max:
                calc_max = dfr[source]
        else:
            calc_max = calc_spt

        if is_crossed == True:
            if dfr[source] < calc_min:
               calc_min = dfr[source]
        elif calc_os == 0:
            if dfr[source] < calc_min:
               calc_min = dfr[source]
        else:
            
            calc_min = calc_spt
        
        avg = (calc_max + calc_min)/2
        calc_source = dfr[source]
        return avg, calc_min,calc_max,calc_spt,calc_os
    calcFt(None, init=1)
    dtfT[['calc_avg','calc_min', 'calc_max','calc_spt','calc_os']] = dtfT.apply(calcFt, axis = 1, result_type='expand')
    return dtfT['calc_avg']


class SuperIchi(IStrategy):
   
    INTERFACE_VERSION = 2
    minimal_roi = {
        "0": 0.999
    }
    stoploss = -0.99

    # Trailing stoploss
    trailing_stop = False


    # Buy hyperspace params:
    buy_params = {
        "kijun_len_buy": 7,
        "kijun_mult_buy": 3,
        "tenkan_len_buy": 22,
        "tenkan_mult_buy": 1,
    }

    # Sell hyperspace params:
    sell_params = {
        "kijun_len_sell": 23,
        "kijun_mult_sell": 5,
        "tenkan_len_sell": 10,
        "tenkan_mult_sell": 3,
    }

    tenkan_len_buy = IntParameter(1, 30, default=buy_params['tenkan_len_buy'], space='buy', optimize=True)
    tenkan_mult_buy = IntParameter(1, 6, default=buy_params['tenkan_mult_buy'], space='buy', optimize=True)
    kijun_len_buy = IntParameter(1, 30, default=buy_params['kijun_len_buy'], space='buy', optimize=True)
    kijun_mult_buy = IntParameter(1, 6, default=buy_params['kijun_mult_buy'], space='buy', optimize=True)

    tenkan_len_sell = IntParameter(1, 30, default=sell_params['tenkan_len_sell'], space='sell', optimize=True)
    tenkan_mult_sell = IntParameter(1, 6, default=sell_params['tenkan_mult_sell'], space='sell', optimize=True)
    kijun_len_sell = IntParameter(1, 30, default=sell_params['kijun_len_sell'], space='sell', optimize=True)
    kijun_mult_sell = IntParameter(1, 6, default=sell_params['kijun_mult_sell'], space='sell', optimize=True)

    # Optimal timeframe for the strategy.
    timeframe = '5m'
    custom_3c_pairs = {}
    custom_main = {}

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

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
            'tenkan_b': {'color': 'red'},
            'kijun_b': {'color': 'blue'},
            'tenkan_s': {'color': 'yellow'},
            'kijun_s': {'color': 'black'} 
        },
        'subplots': {
          
        }
    }




    def informative_pairs(self):

        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

       if self.config['runmode'].value != 'hyperopt':
            dataframe['tenkan_b'] = funcSuperIchi(dataframe, source = 'close', length = self.tenkan_len_buy.value, mult = self.tenkan_mult_buy.value)
            dataframe['kijun_b'] = funcSuperIchi(dataframe, source = 'close', length = self.kijun_len_buy.value, mult = self.kijun_mult_buy.value)
            dataframe['Buy_s'] = np.where( ((dataframe['tenkan_b'] > dataframe['kijun_b']) &  (dataframe['tenkan_b'].shift(1) < dataframe['kijun_b'].shift(1))),1,0)
           
            dataframe['tenkan_s'] = funcSuperIchi(dataframe, source = 'close', length = self.tenkan_len_sell.value, mult = self.tenkan_mult_sell.value)
            dataframe['kijun_s'] = funcSuperIchi(dataframe, source = 'close', length = self.kijun_len_sell.value, mult = self.kijun_mult_sell.value)
            dataframe['Sell_s'] = np.where( ((dataframe['tenkan_s'] < dataframe['kijun_s']) &  (dataframe['tenkan_s'].shift(1) > dataframe['kijun_s'].shift(1))),1,0)

       return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        if self.config['runmode'].value == 'hyperopt':
            dataframe['tenkan_b'] = funcSuperIchi(dataframe, source = 'close', length = self.tenkan_len_buy.value, mult = self.tenkan_mult_buy.value)
            dataframe['kijun_b'] = funcSuperIchi(dataframe, source = 'close', length = self.kijun_len_buy.value, mult = self.kijun_mult_buy.value)
            dataframe['Buy_s'] = np.where( ((dataframe['tenkan_b'] > dataframe['kijun_b']) &  (dataframe['tenkan_b'].shift(1) < dataframe['kijun_b'].shift(1))),1,0)
       
        dataframe.loc[
            (
                ((dataframe['Buy_s'] == 1 )) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

         if self.config['runmode'].value == 'hyperopt':
            dataframe['tenkan_s'] = funcSuperIchi(dataframe, source = 'close', length = self.tenkan_len_sell.value, mult = self.tenkan_mult_sell.value)
            dataframe['kijun_s'] = funcSuperIchi(dataframe, source = 'close', length = self.kijun_len_sell.value, mult = self.kijun_mult_sell.value)
            dataframe['Sell_s'] = np.where( ((dataframe['tenkan_s'] < dataframe['kijun_s']) &  (dataframe['tenkan_s'].shift(1) > dataframe['kijun_s'].shift(1))),1,0)

         dataframe.loc[
            (
            (dataframe['Sell_s'] == 1 ) &
            (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
         return dataframe

 
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import math

from freqtrade.persistence import Trade, PairLocks
from datetime import datetime, timedelta
import math
import logging


from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)
 

 

def funcSMMA(dtloc, source = 'close', length = 14):
    """
    // 
      https://www.tradingview.com/script/QiLlu8z0-Futures-Trading-by-prakash-patel/
     :return: bull and bear   
     translated for freqtrade:
            discord freqtrade: viksal1982  
            email:             viktors.s@gmail.com 
            github:            https://github.com/viktors1982/trading 
    """ 
    col_sma  = 'funcSMMA_sma_col'
    col_ssma = 'funcSMMA_ssma_col'
    dtSMMA = dtloc.copy()
    dtSMMA[col_sma] = ta.SMA(dtSMMA, timeperiod = length)
    def calc_SMMA(dfr, init=0):
        global calc_SMMA_value
        if init == 1:
            calc_SMMA_value = 0
            return
        if calc_SMMA_value == 0 or calc_SMMA_value != calc_SMMA_value:
            calc_SMMA_value = dfr[col_sma]
        calc_SMMA_value = (calc_SMMA_value * (length - 1) + dfr[source]) / length
        return calc_SMMA_value
    calc_SMMA(None, init=1)
    dtSMMA[col_ssma] = dtSMMA.apply(calc_SMMA, axis = 1)
    return dtSMMA[col_ssma]





class FuturesTradingbyPrakashPatel(IStrategy):

    INTERFACE_VERSION = 3



    buy_params = {
        "ssma_jawLength" : 13,
        "ssma_teethLength" : 8,
        "ssma_lipsLength" : 5,

    }

   # ROI table:
    minimal_roi = {
        "0": 0.20,  # This is 10000%, which basically disables ROI
    }
 
    

    

    ssma_jawLength   = IntParameter(1, 100, default= int(buy_params['ssma_jawLength']), space='buy')
    ssma_teethLength = IntParameter(1, 100, default= int(buy_params['ssma_teethLength']), space='buy')
    ssma_lipsLength  = IntParameter(1, 100, default= int(buy_params['ssma_lipsLength']), space='buy')
   
    stoploss = -0.15

    # Trailing stoploss
    trailing_stop = False
   

    timeframe = '5m'
    custom_info = {}
  
    process_only_new_candles = False

  
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

   
    startup_candle_count: int = 30

    can_short = True

    # Optional order type mapping.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        "entry": "gtc",
        "exit": "gtc",
    }
    
    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
           'jaw': {'color': 'green'},
             'teet': {'color': 'blue'},
             'lips': {'color': 'yellow'},

        },
        'subplots': {
         
           
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        

        dataframe['jaw']  = funcSMMA(dataframe, source = 'close', length = int(self.ssma_jawLength.value)) 
        dataframe['teet'] = funcSMMA(dataframe, source = 'close', length = int(self.ssma_teethLength.value)) 
        dataframe['lips'] = funcSMMA(dataframe, source = 'close', length = int(self.ssma_lipsLength.value)) 
        dataframe.to_csv('aaaa.csv')
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
              
                
                (qtpylib.crossed_above(dataframe['jaw'], dataframe['lips'])) &
                (dataframe['volume'] > 0)  
            ),
             ['enter_long', 'enter_tag']] = (1, 'long')


        dataframe.loc[
            (
                 
                (qtpylib.crossed_above(dataframe['lips'], dataframe['jaw'])) &
                (dataframe['volume'] > 0)  
            ),
             ['enter_short', 'enter_tag']] = (1, 'short')


        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                  (qtpylib.crossed_above(dataframe['jaw'], dataframe['lips'])) &
                (dataframe['volume'] > 0)  
            ),
             ['exit_short', 'exit_tag']] = (1, 'short')


        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['lips'], dataframe['jaw'])) &
                (dataframe['volume'] > 0)  
            ),
              ['exit_long', 'exit_tag']] = (1, 'long')



        return dataframe
    


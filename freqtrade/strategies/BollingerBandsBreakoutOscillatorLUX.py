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
 

 

def funcBollingerBandsBreakoutOscillatorLUX(dtloc, source = 'close', length = 14, mult = 1.0):
    """
    // This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/
    // Nadaraya-Watson Envelope [LUX]
      https://www.tradingview.com/script/YaniRMVC-Bollinger-Bands-Breakout-Oscillator-LUX/
     :return: bull and bear   
     translated for freqtrade:
            discord freqtrade: viksal1982  
            email:             viktors.s@gmail.com 
            github:            https://github.com/viktors1982/trading 
    """ 
    dtBBB = dtloc.copy()
    dtBBB['bull']       = np.nan
    dtBBB['bear']       =  np.nan
    dtBBB['stdev']      =  ta.STDDEV(dtBBB, length) * mult
    dtBBB['ema']        =  ta.EMA(dtBBB, length)
    dtBBB['upper']      =  dtBBB['ema'] + dtBBB['stdev']
    dtBBB['lower']      =  dtBBB['ema'] - dtBBB['stdev']
    dtBBB['bull_m']     = pd.DataFrame(np.where(((dtBBB[source] - dtBBB['upper']) < 0), 0, (dtBBB[source] - dtBBB['upper']) )).rolling(length).sum()
    dtBBB['bear_m']     = pd.DataFrame(np.where(((dtBBB['lower'] - dtBBB[source]) < 0), 0, (dtBBB['lower'] - dtBBB[source]) )).rolling(length).sum()
    dtBBB['bull_den']   = (dtBBB[source] - dtBBB['upper']).abs().rolling(length).sum()
    dtBBB['bear_den']   = (dtBBB['lower'] - dtBBB[source]).abs().rolling(length).sum()
    dtBBB['bull']       = dtBBB['bull_m']/dtBBB['bull_den'] * 100
    dtBBB['bear']       = dtBBB['bear_m']/dtBBB['bear_den'] * 100
    return              dtBBB[['bull','bear']]



class BollingerBandsBreakoutOscillatorLUX(IStrategy):

    INTERFACE_VERSION = 3



    buy_params = {
        "bbb_length" : 14,
        "bbb_mult" : 1.0,
        "bbb_enter_limit" : 90

    }

   # ROI table:
    minimal_roi = {
        "0": 0.015,  # This is 10000%, which basically disables ROI
    }
 
    

    

    bbb_length = IntParameter(10, 100, default= int(buy_params['bbb_length']), space='buy')
    bbb_mult = DecimalParameter(0.5, 5, default= float(buy_params['bbb_mult']), space='buy')
    # bbb_enter_limit = IntParameter(50, 100, default= int(buy_params['bbb_enter_limit']), space='buy')
    
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
           
        },
        'subplots': {
            'OSC': {
            'bull': {'color': 'green'},
             'bear': {'color': 'blue'},
        }
           
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe[['bull','bear']] = funcBollingerBandsBreakoutOscillatorLUX(dataframe, source = 'close', length = int(self.bbb_length.value), mult = float(self.bbb_mult.value)) 
        
        dataframe.loc[
            (
              
                
                (dataframe['bull'] > dataframe['bear']) &
                ( dataframe['bull'].shift(1) < dataframe['bear'].shift(1) ) &
                (dataframe['volume'] > 0)  
            ),
             ['enter_long', 'enter_tag']] = (1, 'long')


        dataframe.loc[
            (
                 
                (dataframe['bull'] < dataframe['bear']) &
                (dataframe['bull'].shift(1) > dataframe['bear'].shift(1)) &
                (dataframe['volume'] > 0)  
            ),
             ['enter_short', 'enter_tag']] = (1, 'short')


        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                
                # (dataframe['bull'] > int(self.bbb_enter_limit.value)) &
                # (dataframe['volume'] > 0)  
            ),
             ['exit_short', 'exit_tag']] = (1, 'short')


        dataframe.loc[
            (
            #    (dataframe['bear'] > int(self.bbb_enter_limit.value)) &
            #    (dataframe['volume'] > 0)  
            ),
              ['exit_long', 'exit_tag']] = (1, 'long')



        return dataframe
    


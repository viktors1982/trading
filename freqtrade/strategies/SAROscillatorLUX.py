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
import pandas_ta as pta
import math

from freqtrade.persistence import Trade, PairLocks
from datetime import datetime, timedelta
import math
import logging


from datetime import datetime, timedelta, timezone
#from py3cw.request import Py3CW

logger = logging.getLogger(__name__)
 


class SAROscillatorLUX(IStrategy):

    INTERFACE_VERSION = 3



    buy_params = {
        "acc_buy" : 0.01,
        "inc_buy" : 0.05,
        "lim_buy" : 0.2,

    }

   # ROI table:
    minimal_roi = {
        "0": 0.999,  # This is 10000%, which basically disables ROI
    }
 
   

    acc_buy = DecimalParameter(0.0001, 0.05, default=buy_params['acc_buy'], space='buy', optimize=True)
    inc_buy = DecimalParameter(0.0001, 0.05, default=buy_params['inc_buy'], space='buy', optimize=True)
    lim_buy = DecimalParameter(0.001, 1, default=buy_params['lim_buy'], space='buy', optimize=True)

    
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
            'sar': {'color': 'green'},
           
        },
        'subplots': {
            # Subplots - each dict defines one additional plot
            "OSC": {
                'sosc': {'color': 'blue'},
                'posc': {'color': 'orange'}
            } 
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # // This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/
        # // © LuxAlgo
        # https://www.tradingview.com/script/uc5DGskn-Parabolic-SAR-Oscillator-LUX/
        # translated for freqtrade: viksal1982  viktors.s@gmail.com
      

        acc = float(self.acc_buy.value)
        inc = float(self.inc_buy.value)
        lim = float(self.lim_buy.value)
        df = pta.psar(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], af0=acc, af=inc, max_af=lim)
        dataframe['sar'] = np.where( df[f'PSARl_{acc}_{lim}'] > 0, df[f'PSARl_{acc}_{lim}'] , df[f'PSARs_{acc}_{lim}'] )
        dataframe['cross'] = np.where( ( ( (dataframe['sar'] >  dataframe['close'])  & (dataframe['sar'].shift(1) <  dataframe['close'].shift(1)))  | ((dataframe['sar'] <  dataframe['close'])  & (dataframe['sar'].shift(1) >  dataframe['close'].shift(1))) ) ,1,0)
        dataframe.to_csv('test.csv')
        def calc_max(dfr, init=0):
            global calc_max_value
            if init == 1:
                calc_max_value = 0.0
                return
            if dfr['cross'] == 1:
                if dfr['high'] > dfr['sar']:
                    calc_max_value = dfr['high']
                else:
                    calc_max_value = dfr['sar']
            else:
                if dfr['high'] > calc_max_value:
                    calc_max_value = dfr['high']
            return calc_max_value
        calc_max(None, init=1)
        dataframe['max'] = dataframe.apply(calc_max, axis = 1)

        def calc_min(dfr, init=0):
            global calc_min_value
            if init == 1:
                calc_min_value = 0.0
                return
            if dfr['cross'] == 1:
                if dfr['low'] < dfr['sar']:
                    calc_min_value = dfr['low']
                else:
                    calc_min_value = dfr['sar']
            else:
                if dfr['low'] < calc_min_value:
                    calc_min_value = dfr['low']
            return calc_min_value
        calc_min(None, init=1)
        dataframe['min'] = dataframe.apply(calc_min, axis = 1)


        dataframe['posc'] = (( dataframe['close'] -  dataframe['sar'])/( dataframe['max'] -  dataframe['min'] ) * 100) * -1
        dataframe['sosc'] = ((( dataframe['sar'] -  dataframe['min'])/( dataframe['max'] -  dataframe['min'] ) - 0.5) * -200) *-1


        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
 
        dataframe.loc[
            (
    
                (dataframe['sosc'] == 100) &
                (dataframe['volume'] > 0)  
            ),
             ['enter_long', 'enter_tag']] = (1, 'long')


        dataframe.loc[
            (
                 
                 (dataframe['sosc'] == -100) & 
                (dataframe['volume'] > 0)  
            ),
             ['enter_short', 'enter_tag']] = (1, 'short')


        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        

        dataframe.loc[
            (
                
                (dataframe['sosc'] == 100) &
                (dataframe['volume'] > 0)  
            ),
             ['exit_short', 'exit_tag']] = (1, 'short')


        dataframe.loc[
            (
                  (dataframe['sosc'] == -100) & 
                (dataframe['volume'] > 0)  
            ),
              ['exit_long', 'exit_tag']] = (1, 'long')



        return dataframe
    


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
"""
    https://tr.tradingview.com/script/SlCjQY3v/
    translated for freqtrade: viksal1982  viktors.s@gmail.com
"""

def SuperSM(dataframe, p = 'close', len = 8, len2 = 13, len3 = 3):
        df = dataframe.copy()
       

        f = (1.414*3.14159)/len
        a = math.exp(-f)
        c2 = 2*a*math.cos(f)
        c3 = -a*a
        c1 = 1-c2-c3


        def calc_ssmooth(dfr, init=0):
            global calc_ssmooth_value
            global calc_src_value
            if init == 1:
                calc_ssmooth_value = [0.0] * 2
                calc_src_value = [0.0] * 2
                return
            calc_src_value.pop(0)
            calc_src_value.append(dfr[p])
            ssm =  c1*(calc_src_value[-1]+calc_src_value[-2])*0.5+c2*(calc_ssmooth_value[-1])+c3*(calc_ssmooth_value[-2])       
            calc_ssmooth_value.pop(0)
            calc_ssmooth_value.append(ssm)
            return ssm
        calc_ssmooth(None, init=1)
        df['ssmooth'] = df.apply(calc_ssmooth, axis = 1)

       
        f2 = (1.414*3.14159)/len2
        a2 = math.exp(-f2)
        c22 = 2*a2*math.cos(f2)
        c32 = -a2*a2
        c12 = 1-c22-c32
        def calc_ssmooth2(dfr, init=0):
            global calc_ssmooth_value
            global calc_src_value
            if init == 1:
                calc_ssmooth_value = [0.0] * 2
                calc_src_value = [0.0] * 2
                return
            calc_src_value.pop(0)
            calc_src_value.append(dfr[p])
            ssm =  c12*(calc_src_value[-1]+calc_src_value[-2])*0.5+c22*(calc_ssmooth_value[-1])+c32*(calc_ssmooth_value[-2])       
            calc_ssmooth_value.pop(0)
            calc_ssmooth_value.append(ssm)
            return ssm
        calc_ssmooth2(None, init=1)
        df['ssmooth2'] = df.apply(calc_ssmooth2, axis = 1)

        df['macd'] = (df['ssmooth'] - df['ssmooth2'])*10000000
        f3 = (1.414*3.14159)/len3
        a3 = math.exp(-f3)
        c23 = 2*a3*math.cos(f3)
        c33 = -a3*a3
        c13 = 1-c23-c33
        def calc_ssmooth3(dfr, init=0):
            global calc_ssmooth_value
            global calc_src_value
            if init == 1:
                calc_ssmooth_value = [0.0] * 2
                calc_src_value = [0.0] * 2
                return
            calc_src_value.pop(0)
            calc_src_value.append(dfr['macd'])
            ssm =  c13*(calc_src_value[-1]+calc_src_value[-2])*0.5+c23*(calc_ssmooth_value[-1])+c33*(calc_ssmooth_value[-2])       
            calc_ssmooth_value.pop(0)
            calc_ssmooth_value.append(ssm)
            return ssm
        calc_ssmooth3(None, init=1)
        df['ssmooth3'] = df.apply(calc_ssmooth3, axis = 1)

        return df['ssmooth3'], df['macd']

class SuperSmoothedMACDforCRYPTO(IStrategy):
  
    p1_buy = IntParameter(1, 100, default= 8, space='buy')
    p2_buy = IntParameter(1, 100, default= 13, space='buy')
    p3_buy = IntParameter(1, 100, default= 3, space='buy')

    p1_sell = IntParameter(1, 100, default= 8, space='sell')
    p2_sell = IntParameter(1, 100, default= 13, space='sell')
    p3_sell = IntParameter(1, 100, default= 3, space='sell')
    INTERFACE_VERSION = 2


    stoploss = -0.99



    # Trailing stoploss
    trailing_stop = False


    # Optimal timeframe for the strategy.
    timeframe = '5m'

    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

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
        },
        'subplots': {
            # Subplots - each dict defines one additional plot
            "OSC": {
                'macd': {'color': 'blue'},
                'ssmooth3': {'color': 'orange'},
            }
        }
    }



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        
        # dataframe.to_csv('aaa.csv')
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        p = 'close'
        len = self.p1_buy.value
        len2 = self.p2_buy.value
        len3 = self.p3_buy.value
        dataframe['ssmooth3_buy'], dataframe['macd_buy']  = SuperSM(dataframe,p,len,len2,len3)
        
        dataframe.loc[
            (
                
              
                 (qtpylib.crossed_above(dataframe['ssmooth3_buy'], dataframe['macd_buy'])) &  
                 (dataframe['ssmooth3_buy'] > dataframe['ssmooth3_buy'].shift(1)) &
                 (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        p = 'close'
        len = self.p1_sell.value
        len2 = self.p2_sell.value
        len3 = self.p3_sell.value

        dataframe['ssmooth3_sell'], dataframe['macd_sell']  = SuperSM(dataframe,p,len,len2,len3)

        dataframe.loc[
            (

                (qtpylib.crossed_above(dataframe['macd_sell'], dataframe['ssmooth3_sell'])) &  
                (dataframe['ssmooth3_sell'] < dataframe['ssmooth3_sell'].shift(1)) & 
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
    
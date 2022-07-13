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
#from py3cw.request import Py3CW

logger = logging.getLogger(__name__)
 


def funcLinearRegressionChannel2(dtloc, source = 'close', window = 180, deviations = 2):
    dtLRC = dtloc.copy()
    dtLRC['lrc_up'] = np.nan
    dtLRC['lrc_down'] = np.nan
    dtLRC['slope'] = np.nan
    dtLRC['average'] = np.nan
    dtLRC['intercept'] = np.nan
    i = np.arange(start=1, stop=window+1)
    # print(i)
    i = i[::-1]
    # print(i)
    
    i = i + 1.0
    # print(i)
    Ex = i.sum()
    Ex2 = (i * i).sum()
    ExT2 = math.pow(Ex, 2)
    def calc_lrc(dfr, init=0):
        global calc_lrc_src_value
        if init == 1:
            calc_lrc_src_value = list()
            return
        calc_lrc_src_value.append(dfr[source])
        lrc_val_up = np.nan
        lrc_val_down = np.nan
        slope = np.nan
        average = np.nan
        intercept = np.nan
        Ey  = np.nan
        Exy = np.nan
        vwap1 = np.nan
        sdev = np.nan
        dev = np.nan
        lrc_down = np.nan
        lrc_up = np.nan
        if len(calc_lrc_src_value) > window:
            calc_lrc_src_value.pop(0)
        if len(calc_lrc_src_value) >= window:
            src1 = np.array(calc_lrc_src_value)
            src = src1[::-1]
            Ey  = src.sum()
            # Ey2 = (src * src).sum()
            EyT2 = math.pow(Ey,2)
            Exyi = i*src
            Exy = (Exyi).sum()
            # PearsonsR = (Exy - Ex * Ey / window) / (math.sqrt(Ex2 - ExT2 / window) * math.sqrt(Ey2 - EyT2 / window))
            ExEx = Ex * Ex
            slope = 0.0
            if (Ex2 != ExEx ):
                slope = (window * Exy - Ex * Ey) / (window * Ex2 - ExEx)

            average = Ey / window
            intercept = average - slope * Ex / window + slope
            vwap1 = intercept + slope * window
            sdev = np.std(src1)
            dev = deviations * sdev
            lrc_down = vwap1 - dev
            lrc_up = vwap1 + dev
           
        return slope,average,intercept,Ex,Ey,Ex2,Exy,vwap1,sdev,dev,lrc_down,lrc_up
    calc_lrc(None, init=1)
    dtLRC[['slope','average','intercept','Ex','Ey','Ex2','Exy','vwap1','sdev','dev','lrc_down','lrc_up']] = dtLRC.apply(calc_lrc, axis = 1, result_type='expand')
    return dtLRC[['slope','average','intercept','Ex','Ey','Ex2','Exy','vwap1','sdev','dev','lrc_down','lrc_up']]



class SupertrendBFutures(IStrategy):

    INTERFACE_VERSION = 3


    buy_params = {
        "lr_win" : 150,
        "lr_mult" : 2,

    }

   # ROI table:
    minimal_roi = {
        "0": 0.015,  # This is 10000%, which basically disables ROI
    }
 
 



    lr_win = IntParameter(10, 500, default= int(buy_params['lr_win']), space='buy')
    lr_mult = IntParameter(1, 20, default= int(buy_params['lr_mult']), space='buy')

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
            'lrc_up': {'color': 'green'},
            'lrc_down': {'color': 'blue'},
            'st_line': {'color': 'orange'}
        },
        'subplots': {
         
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #    Super trend B 
    #  https://www.tradingview.com/v/veDICWLK/
    #  translated for freqtrade: viksal1982  viktors.s@gmail.com
        
        mult1 = 0.5
        length = 20
        st_mult = 3
        st_period = 7

        dataframe['basis'] = ta.SMA(dataframe, timeperiod = length) 
        dataframe['upper1'] =  dataframe['basis'] + mult1 * ta.STDDEV(dataframe, length)
        dataframe['lower1'] =  dataframe['basis'] - mult1 * ta.STDDEV(dataframe, length)
        dataframe['ATR'] = ta.ATR(dataframe, timeperiod = st_period)
        dataframe['TR'] = ta.TRANGE(dataframe)
        dataframe['ATR1'] = dataframe['TR'].ewm(alpha=1 / st_period).mean()   
        dataframe['up_lev'] = dataframe['upper1'] - st_mult * dataframe['ATR']
        dataframe['dn_lev'] = dataframe['lower1'] + st_mult * dataframe['ATR']
        dataframe['up_trend'] = dataframe['up_lev']
        dataframe['down_trend'] = dataframe['up_lev']
        dataframe['trend'] = np.where( dataframe['close'] > dataframe['down_trend'], 1, -1)
        dataframe['st_line'] = np.where( dataframe['trend'] == 1, dataframe['up_trend'],dataframe['down_trend'])
         

        dataframe[['slope','average','intercept','Ex','Ey','Ex2','Exy','vwap1','sdev','dev','lrc_down','lrc_up']] = funcLinearRegressionChannel2(dataframe, source = 'close', window = int(self.lr_win.value), deviations = int(self.lr_mult.value)) 

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
    
        dataframe.loc[
            (
                
              
                (qtpylib.crossed_below(dataframe['close'], dataframe['lrc_down'])) &
                (dataframe['volume'] > 0)  
            ),
             ['enter_long', 'enter_tag']] = (1, 'long')


        dataframe.loc[
            (
                 
                
                (qtpylib.crossed_below(dataframe['close'], dataframe['st_line'])) &
                (dataframe['volume'] > 0)  
            ),
             ['enter_short', 'enter_tag']] = (1, 'short')


        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        

       

        dataframe.loc[
            (
                
                 (qtpylib.crossed_below(dataframe['close'], dataframe['lrc_down'])) &
                (dataframe['volume'] > 0)  
            ),
             ['exit_short', 'exit_tag']] = (1, 'short')


        dataframe.loc[
            (
               (qtpylib.crossed_below(dataframe['close'], dataframe['st_line'])) &
                (dataframe['volume'] > 0)  
            ),
              ['exit_long', 'exit_tag']] = (1, 'long')



        return dataframe
    


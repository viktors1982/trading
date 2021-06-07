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
      https://www.tradingview.com/script/WJSDkJwU-Boom-Hunter/ 

      translated for freqtrade: viksal1982  viktors.s@gmail.com


    """



class BoomHunter(IStrategy):


    line1 = DecimalParameter(-1, 0, default=-0.9, space='buy')
    LPPeriod = IntParameter(10, 70, default=20, space='buy')
    K1 = DecimalParameter(0.1, 2, default=0.8, space='buy')
    K2 = DecimalParameter(0.1, 2, default=0.4, space='buy')
    ST = IntParameter(10, 70, default=20, space='buy')
    RS = IntParameter(10, 70, default=20, space='buy')
    smoothK = IntParameter(2, 10, default=3, space='buy')
    smoothD = IntParameter(2, 10, default=3, space='buy')
    ceof_sell = DecimalParameter(0.1, 2, default=0.5, space='sell')

  
    INTERFACE_VERSION = 2


    stoploss = -0.99

  
    trailing_stop = False
 

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
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
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            # Subplots - each dict defines one additional plot
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        


        LPPeriod = int(self.LPPeriod.value)
        K1 = float(self.K1.value)
        K2 = float(self.K2.value)

        #Stoch RSI
        smoothK =  int(self.smoothK.value)
        smoothD = int(self.smoothD.value)
        lengthRSI = int(self.RS.value)
        lengthStoch = int(self.ST.value)
        src9 = 'close'
        dataframe['rsi1']  = ta.RSI(dataframe['close'], timeperiod = lengthRSI)

        dataframe['stoch'] =  100 * ((dataframe['rsi1'] - dataframe['rsi1'].rolling(lengthStoch).min()) / ( dataframe['rsi1'].rolling(lengthStoch).max()  - dataframe['rsi1'].rolling(lengthStoch).min()))

        dataframe['k9'] = ta.SMA(dataframe['stoch'], timeperiod = smoothK)
        dataframe['d9'] = ta.SMA(dataframe['k9'], timeperiod =  smoothD)

        alpha1 = 0.00 
        HP = 0.00 
        a1 = 0.00 
        b1 = 0.00 
        c1 = 0.00 
        c2 = 0.00 
        c3 = 0.00 
        Filt = 0.00 
        Peak = 0.00
        X = 0.00 
        Quotient1 = 0.00 
        Quotient2 = 0.00
        pi = 2 * math.asin(1)

        alpha1 = ( math.cos( .707 * 2 * pi / 100 ) + math.sin( .707 * 2 * pi / 100 ) - 1 ) / math.cos( .707 * 2 * pi / 100 ) 

        def calc_HP(dfr, init=0):
            global calc_HP_close
            global calc_HP_value
            if init == 1:
                calc_HP_value = list([0.0, 0.0, 0.0])
                calc_HP_close = list([0.0, 0.0, 0.0])
                return

            calc_HP_close.pop(0)
            calc_HP_close.append(dfr['close'])
            hp_val = 0.0
            hp_val = ( 1 - alpha1 / 2 ) * ( 1 - alpha1 / 2 ) * ( calc_HP_close[-1] - 2 * calc_HP_close[-2] + calc_HP_close[-3] ) + 2 * ( 1 - alpha1 ) * calc_HP_value[-1]  - ( 1 - alpha1 ) * ( 1 - alpha1 ) * calc_HP_value[-2]
            calc_HP_value.pop(0)
            calc_HP_value.append(hp_val)
            
            return hp_val
        calc_HP(None, init=1)
        dataframe['HP'] = dataframe.apply(calc_HP, axis = 1)

        
        a1  = math.exp( -1.414 * pi / LPPeriod ) 
        b1  = 2 * a1 * math.cos( 1.414* pi / LPPeriod ) 
        c2  = b1 
        c3  = -a1 * a1 
        c1  = 1 - c2 - c3 

        def calc_Filt(dfr, init=0):
            global calc_Filt_HP
            global calc_Filt_Filt
            if init == 1:
                calc_Filt_HP = list([0.0, 0.0])
                calc_Filt_Filt = list([0.0, 0.0])
                return
            calc_Filt_HP.pop(0)
            calc_Filt_HP.append(dfr['HP'])
            Filt_val =  c1 * ( calc_Filt_HP[-1] + calc_Filt_HP[-2] ) / 2 + c2 * calc_Filt_Filt[-1] + c3 * calc_Filt_Filt[-2] 
            calc_Filt_Filt.pop(0)
            calc_Filt_Filt.append(Filt_val)
            return Filt_val
        calc_Filt(None, init=1)
        dataframe['Filt'] = dataframe.apply(calc_Filt, axis = 1)


        def calc_Peak(dfr, init=0):
            global calc_Peak_Peak
            global calc_Peak_Filt
            if init == 1:
                calc_Peak_Peak = list([0.0])
                calc_Peak_Filt = list([0.0])
                return
            calc_Peak_Filt.pop(0)
            calc_Peak_Filt.append(dfr['Filt'])
            Peak =  .991 * calc_Peak_Peak[-1]
            if abs(calc_Peak_Filt[-1]) > Peak:
                Peak = abs(calc_Peak_Filt[-1])
            calc_Peak_Peak.pop(0)
            calc_Peak_Peak.append(Peak)
            return Peak
        calc_Peak(None, init=1)
        dataframe['Peak'] = dataframe.apply(calc_Peak, axis = 1)

        dataframe['X'] = np.where(dataframe['Peak'] != 0, dataframe['Filt']/dataframe['Peak'], 0)
        dataframe['Quotient1'] = (dataframe['X'] + K1) / ( K1 * dataframe['X'] + 1 )
        dataframe['Quotient2'] = (dataframe['X'] + K2) / ( K2 * dataframe['X'] + 1 )

        dataframe['ema1'] = ta.EMA(dataframe, timeperiod = 30)
        dataframe['sma10'] = ta.SMA(dataframe, timeperiod = 10)
        dataframe['sma30'] = ta.SMA(dataframe, timeperiod = 30)
        dataframe['sma200'] = ta.SMA(dataframe,timeperiod = 200)
        dataframe['ema200'] = ta.EMA(dataframe,timeperiod = 200)

        dataframe.to_csv('test.csv')

       

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
     
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['Quotient2'], float(self.line1.value))) &   
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
   
        dataframe.loc[
            (
                
                (dataframe['Quotient1'] > dataframe['Quotient2']) &  
                 (dataframe['close'] > dataframe['sma200']) &   
                (dataframe['Quotient1'] > float(self.ceof_sell.value)) &   
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
    
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union
from freqtrade.persistence import Trade

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
from  datetime import timedelta
import math

class dip_catcher(IStrategy):
    """
    // 
    // Dip / Retracement Catcher
      https://github.com/Haehnchen/crypto-trading-bot/tree/master/src/modules/strategy/strategies/dip_catcher
      https://github.com/Haehnchen/crypto-trading-bot/blob/master/src/modules/strategy/strategies/dip_catcher/dip_catcher.js
       translated for freqtrade: viksal1982  viktors.s@gmail.com  
       https://github.com/viktors1982/trading/tree/main/freqtrade/strategies
    """ 

 
    INTERFACE_VERSION = 3

    timeframe = '5m'

    # Can this strategy go short?
    can_short: bool = True

    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.03
    }

    stoploss = -0.08
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30


    length_hma = IntParameter(4, 20, default=9, space="buy", optimize=True)
    length_hma_high = IntParameter(4, 20, default=9, space="buy", optimize=True)
    length_hma_low = IntParameter(4, 20, default=9, space="buy", optimize=True)
    bollinger_window = IntParameter(2, 40, default=20, space="buy", optimize=True)
    trend_cloud_multiplier = IntParameter(1, 10, default=4, space="buy", optimize=True)


    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }
    
    @property
    def plot_config(self):
        return {
            'main_plot': {

            },
            'subplots': {
            }
        }

    def informative_pairs(self):

        return []
    
    def ichimoku_cloud(self, dataframe, conversion_periods=9, base_periods=26, lagging_span2_periods=52, displacement=26):
        
        def donchian_channel(series, length):
            return (series.rolling(length).min() + series.rolling(length).max()) / 2
        dataframe['conversion_line'] = (donchian_channel(dataframe['high'], conversion_periods) + donchian_channel(dataframe['low'], conversion_periods)) / 2
        dataframe['base_line'] = (donchian_channel(dataframe['high'], base_periods) + donchian_channel(dataframe['low'], base_periods)) / 2
        dataframe['lead_line1'] = ((dataframe['conversion_line'] + dataframe['base_line']) / 2).shift(displacement)
        dataframe['lead_line2'] = ((donchian_channel(dataframe['high'], lagging_span2_periods) + donchian_channel(dataframe['low'], lagging_span2_periods)) / 2).shift(displacement)
        dataframe['lagging_span'] = dataframe['close'].shift(-displacement)
        return dataframe[['conversion_line', 'base_line', 'lead_line1', 'lead_line2', 'lagging_span']]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe
    

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
       
        
        dataframe['hma'] = ta.WMA(
                    2 * ta.WMA(dataframe['close'], int(math.floor(int(self.length_hma.value)/2))) - ta.WMA(dataframe['close'], int(self.length_hma.value)), int(round(np.sqrt(int(self.length_hma.value))))
                    )
     
        dataframe['hma_high'] = ta.WMA(
                    2 * ta.WMA(dataframe['close'], int(math.floor(int(self.length_hma_high.value/2)))) - ta.WMA(dataframe['close'], int(self.length_hma_high.value)), int(round(np.sqrt(int(self.length_hma_high.value))))
                    )
 
        dataframe['hma_low'] = ta.WMA(
                    2 * ta.WMA(dataframe['close'], int(math.floor(int(self.length_hma_low.value/2)))) - ta.WMA(dataframe['close'], int(self.length_hma_low.value)), int(round(np.sqrt(int(self.length_hma_low.value))))
                    )
        

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=int(self.bollinger_window.value), stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
 
        dataframe[['conversion_line', 'base_line', 'lead_line1', 'lead_line2', 'senkouspanb']] = self.ichimoku_cloud(dataframe, conversion_periods= (9 * int(self.trend_cloud_multiplier.value)), base_periods=(26 * int(self.trend_cloud_multiplier.value)) , lagging_span2_periods=(52* int(self.trend_cloud_multiplier.value)), displacement=(10* int(self.trend_cloud_multiplier.value)))


        dataframe.loc[
            (
                # Condition for a long entry
                (dataframe['hma_low'] > dataframe['bb_lowerband']) &
                (dataframe['hma'] > dataframe['senkouspanb'])   
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                # Condition for a short entry
                (dataframe['hma_high'] < dataframe['bb_upperband']) &
                (dataframe['hma'] < dataframe['senkouspanb'])   
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
       
        return dataframe
    
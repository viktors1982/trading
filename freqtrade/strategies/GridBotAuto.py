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


from py3cw.request import Py3CW

logger = logging.getLogger(__name__)



def funcGridBotAuto(dtloc, source = 'close', iLen = 7, iGrids = 6, iMA = 'sma', iLZ = 0.35, iELSTX = 15.0, iGI = 0.06, iEXTR = True, iDir = 'neutral', iReset = True):
   
    G = iGrids
    iLZ = iLZ / 100
    iGI = iGI / 100
    dtloc['LR']   = ta.LINEARREG(dtloc[source], timeperiod=iLen)
    dtloc['SMA']  = ta.SMA(dtloc[source], timeperiod=iLen)
    dtloc['EMA']  =  ta.EMA(dtloc[source], timeperiod=iLen)
    # dtloc['VWMA'] = ta.VWMA(dtloc[source], timeperiod=iLen)
    dtloc['TEMA'] = ta.EMA(ta.EMA(ta.EMA(dtloc[source], timeperiod=iLen), timeperiod=iLen), timeperiod=iLen)
    if iMA == 'lreg':
        dtloc['MA'] = dtloc['LR']
    elif iMA == 'sma':
        dtloc['MA'] = dtloc['SMA']
    elif iMA == 'ema':
        dtloc['MA'] = dtloc['EMA']
    elif iMA == 'vwma':
        dtloc['MA'] = dtloc['VWMA']
    else:
        dtloc['MA'] = dtloc['TEMA']
    def calc_lz(dfr, init=0):
        global calc_lza_value
        global calc_x_value
        if init == 1:
            calc_lza_value = 0.0
            calc_x_value   = 0.0
            return
        s = 0.0
        if dfr['MA'] > 0:
            s = 1.0
        elif dfr['MA'] < 0:
            s = -1.0
        if dfr['MA'] == calc_x_value:
            calc_lza_value = dfr['MA']
        elif dfr['MA'] > (calc_lza_value + iLZ * calc_lza_value * s):
            calc_lza_value = dfr['MA']
        elif dfr['MA'] < (calc_lza_value - iLZ * calc_lza_value * s):
            calc_lza_value = dfr['MA']
        calc_x_value = dfr['MA']
        return calc_lza_value
    calc_lz(None, init=1)
    dtloc['LMA'] = dtloc.apply(calc_lz, axis = 1)
    ELSTX = 0.001 * iELSTX  ### TODO 
    def calc_ap_nup_ndown(dfr, init=0):
        global calc_ap_value
        global calc_nextup_value
        global calc_nextdown_value
        global calc_lma_value
        global calc_low_value
        global calc_high_value
        global calc_open_value
        global calc_close_value
        global calc_SignalLine_value
        global calc_LastSignal_value
        global calc_LastSignal_Index_value
        global calc_Sell_s_value
        global calc_Buy_s_value
        if init == 1:
            calc_ap_value = 0.0
            calc_nextup_value   = 0.0
            calc_nextdown_value   = 0.0
            calc_lma_value = 0.0
            calc_low_value = 0.0
            calc_high_value = 0.0
            calc_open_value = 0.0
            calc_close_value = 0.0
            calc_SignalLine_value = 0.0
            calc_Sell_s_value = 0
            calc_Buy_s_value = 0
            calc_LastSignal_value = 0.0
            calc_LastSignal_Index_value = 0
            return
        if dfr['MA'] > dfr['LMA']:
            calc_ap_value = calc_ap_value + ELSTX
        elif dfr['MA'] < dfr['LMA']:
            calc_ap_value = calc_ap_value - ELSTX
        AP1 = calc_ap_value
        if calc_ap_value >= calc_nextup_value:
            calc_ap_value = calc_nextup_value
        if calc_ap_value <= calc_nextdown_value:
            calc_ap_value = calc_nextdown_value
        AP2 = calc_ap_value
        if dfr['LMA'] != calc_lma_value:
            calc_ap_value = dfr['LMA']

        if dfr['LMA'] != calc_lma_value:
            calc_nextup_value = dfr['LMA'] + dfr['LMA'] * iGI

        if dfr['LMA'] != calc_lma_value:
            calc_nextdown_value = dfr['LMA'] - dfr['LMA'] * iGI   

        GI = calc_ap_value * iGI
        a_grid = np.zeros(9)
        for i in range(len(a_grid)):
            a_grid[i] = calc_ap_value + GI * (i - 4)
        
        G0 = a_grid[0]  #Upper4
        G1 = a_grid[1]  #Upper3
        G2 = a_grid[2]  #Upper2
        G3 = a_grid[3]  #Upper1
        G4 = a_grid[4]  #Center
        G5 = a_grid[5]  #Lower1
        G6 = a_grid[6]  #Lower2
        G7 = a_grid[7]  #Lower3
        G8 = a_grid[8]  #Lower4

        UpperLimit = G5
        if G >= 8:
            UpperLimit = G8
        elif G >= 6:
            UpperLimit = G7
        elif G >= 4:
            UpperLimit = G6

        LowerLimit = G3
        if G >= 8:
            LowerLimit = G0
        elif G >= 6:
            LowerLimit = G1
        elif G >= 4:
            LowerLimit = G2
        Value = 0.0
        Buy_Index = 0
        Sell_Index = 0
        start = int(4 - G / 2)
        end = int((4 + G / 2) + 1)
    	
        for i in range(start, end):
            Value = a_grid[i]
            if iEXTR:
                if calc_low_value < Value and dfr['high'] >= Value:
                    Sell_Index = i
                if calc_high_value > Value and dfr['low'] <= Value:
                    Buy_Index = i
            else:
                if calc_close_value < Value and dfr['close'] >= Value:
                    Sell_Index = i
                if calc_close_value > Value and dfr['close'] <= Value:
                    Buy_Index = i


        Buy_s = 0
        if Buy_Index > 0:
            Buy_s = 1
        Sell_s = 0
        if Sell_Index > 0:
            Sell_s = 1

        prevcalc_SignalLine_value = calc_SignalLine_value
        if dfr['low'] >= (calc_SignalLine_value - GI):
            Buy_s = 0

        if dfr['high'] <= (calc_SignalLine_value + GI):
            Sell_s = 0
        
        if dfr['close'] > UpperLimit:
            Buy_s = 0
        if dfr['close'] < LowerLimit:
            Buy_s = 0

        if dfr['close'] < LowerLimit:
            Sell_s = 0
        if dfr['close'] > UpperLimit:
            Sell_s = 0

        DIR = 0 
        if iDir == 'up':
            DIR = 1
        if iDir == 'down':
            DIR = -1

        if DIR == -1 and dfr['low'] >= (calc_SignalLine_value - GI * 2):
            Buy_s = 0
        if DIR == 1 and dfr['high'] <= (calc_SignalLine_value + GI * 2):
            Sell_s = 0

        if Buy_s == 1 and Sell_s == 1:
            Buy_s = 0
            Sell_s = 0
        

        #Cooldown need array --
        if calc_Sell_s_value == 1 or calc_Buy_s_value == 1:
            Buy_s = 0
            Sell_s = 0
        # -------------
        if Buy_s == 1:
            calc_LastSignal_value = 1
            calc_LastSignal_Index_value = Buy_Index
        if Sell_s == 1:
            calc_LastSignal_value = -1
            calc_LastSignal_Index_value = Sell_Index

        calc_SignalLine_value = a_grid[calc_LastSignal_Index_value]
        
        if iReset:
            if dfr['LMA'] < calc_lma_value:
                calc_SignalLine_value = UpperLimit
            if dfr['LMA'] > calc_lma_value:
                calc_SignalLine_value = LowerLimit
        calc_lma_value = dfr['LMA']
        calc_low_value = dfr['low']
        calc_high_value = dfr['high']
        calc_open_value = dfr['open']
        calc_close_value = dfr['close']
        calc_Sell_s_value = Sell_s
        calc_Buy_s_value = Buy_s
        return Buy_s,Sell_s,G0,G1,G2,G3,G4,G5,G6,G7,G8, calc_ap_value,calc_nextup_value, calc_nextdown_value,AP1,AP2,UpperLimit,LowerLimit,Buy_Index,Sell_Index,prevcalc_SignalLine_value,calc_SignalLine_value,
    calc_ap_nup_ndown(None, init=1)
    dtloc[['Buy_s','Sell_s','G0','G1','G2','G3','G4','G5','G6','G7','G8','AP', 'NEXTUP','NEXTDOWN','AP1','AP2','UpperLimit','LowerLimit','Buy_Index','Sell_Index','PrevSignalLine','SignalLine' ]] = dtloc.apply(calc_ap_nup_ndown, axis = 1, result_type='expand')

    

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
    # global wn
    # global sumSCW
    # try:
    #     wn
    # except NameError:
    #     print('i am here')
    wn = np.zeros((window, window))
    for i in range(window):
        for j in range(window):
            wn[i,j] = math.exp(-(math.pow(i-j,2)/(bandwidth*bandwidth*2)))
    sumSCW = wn.sum(axis = 1)
    def calc_nwa(dfr, init=0):
        global calc_nwa_src_value
        if init == 1:
            calc_nwa_src_value = list()
            return
        calc_nwa_src_value.append(dfr[source])
        mae = 0.0
        y2_val = 0.0
        y2_val_up = np.nan
        y2_val_down = np.nan
        if len(calc_nwa_src_value) > window:
            calc_nwa_src_value.pop(0)
        if len(calc_nwa_src_value) >= window:
            src = np.array(calc_nwa_src_value)
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


def funcLinearRegressionChannel2(dtloc, source = 'close', window = 180, deviations = 2):
    dtLRC = dtloc.copy()
    dtLRC['lrc_up'] = np.nan
    dtLRC['lrc_down'] = np.nan
    i = np.arange(window)
    i = i[::-1]
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
        if len(calc_lrc_src_value) > window:
            calc_lrc_src_value.pop(0)
        if len(calc_lrc_src_value) >= window:
            src = np.array(calc_lrc_src_value)
            Ey  = src.sum()
            Ey2 = (src * src).sum()
            EyT2 = math.pow(Ey,2)
            Exy = (i*src).sum()
            PearsonsR = (Exy - Ex * Ey / window) / (math.sqrt(Ex2 - ExT2 / window) * math.sqrt(Ey2 - EyT2 / window))
            ExEx = Ex * Ex
            slope = 0.0
            if (Ex2 != ExEx ):
                slope = (window * Exy - Ex * Ey) / (window * Ex2 - ExEx)
            linearRegression = (Ey - slope * Ex) / window
            intercept = linearRegression + window * slope
            deviation = np.power((src - (intercept - slope * (window - i))), 2).sum()
            devPer = deviation / window
            devPerSqrt = math.sqrt(devPer)
            deviation = deviations * devPerSqrt
            lrc_val_up = linearRegression  + deviation
            lrc_val_down = linearRegression - deviation
        return lrc_val_up,lrc_val_down
    calc_lrc(None, init=1)
    dtLRC[['lrc_up','lrc_down']] = dtLRC.apply(calc_lrc, axis = 1, result_type='expand')
    return dtLRC[['lrc_up','lrc_down']]


def LinearRegressionChannel2(dtloc, source = 'close', window = 180, deviations = 2):
    dtLRC = dtloc.copy()
    dtLRC['lrc_up'] = np.nan
    dtLRC['lrc_down'] = np.nan
    colSource = dtLRC.loc[:, source].values
    collrc_up = dtLRC.loc[:, 'lrc_up'].values
    collrc_down = dtLRC.loc[:, 'lrc_down'].values
    Ex = 0.0
    Ey = 0.0
    Ex2 = 0.0
    Ey2 = 0.0
    Exy = 0.0
    for i in range(window):
        closeI = colSource[-(i+1)]
        Ex = Ex + i 
        Ey = Ey + closeI
        Ex2 = Ex2 + i*i
        Ey2 = Ey2 + closeI*closeI
        Exy = Exy + i*closeI
    ExT2 = math.pow(Ex,2)
    EyT2 = math.pow(Ey,2)
    PearsonsR = (Exy - Ex * Ey / window) / (math.sqrt(Ex2 - ExT2 / window) * math.sqrt(Ey2 - EyT2 / window))
    ExEx = Ex * Ex
    slope = 0.0
    if (Ex2 != ExEx ):
        slope = (window * Exy - Ex * Ey) / (window * Ex2 - ExEx)
    linearRegression = (Ey - slope * Ex) / window
    intercept = linearRegression + window * slope
    deviation = 0.0
    for i in range(window):
        deviation = deviation + math.pow((colSource[-(i+1)] - (intercept - slope * (window - i))), 2)
    devPer = deviation / window
    devPerSqrt = math.sqrt(devPer)
    deviation = deviations * devPerSqrt
    for i in range(window):
        collrc_up[-(i+1)] = (linearRegression + slope * i) + deviation
        collrc_down[-(i+1)] = (linearRegression + slope * i) - deviation
    dtLRC['lrc_up'] = collrc_up.tolist()
    dtLRC['lrc_down'] = collrc_down.tolist()
    return dtLRC[['lrc_up','lrc_down']]

class GridBotAuto(IStrategy):
   
    INTERFACE_VERSION = 2
    minimal_roi = {
        "0": 0.9999
    }
    stoploss = -0.99

    # Trailing stoploss
    trailing_stop = False

    buy_params = {
        "iELSTX_buy": 57,
        "iGI_buy": 4.655,
        "iLZ_buy": 6.54,
        "iLen_buy": 5,
    }

    iLen_buy = IntParameter(2, 15, default=buy_params['iLen_buy'], space='buy', optimize=True)
    iLZ_buy = DecimalParameter(0.01, 10.0, default=buy_params['iLZ_buy'], space='buy', optimize=True)
    iELSTX_buy = IntParameter(5, 75, default=buy_params['iELSTX_buy'], space='buy', optimize=True)
    iGI_buy = DecimalParameter(0.01, 10.0, default=buy_params['iGI_buy'], space='buy', optimize=True)

    window_2_buy = IntParameter(20, 250, default=180, space='buy', optimize=True)
    deviations_buy   = IntParameter(1, 20, default=2, space='buy', optimize=True)

    c3_key    = ''
    c3_secret = ''
    c3_mode   = 'real'
    c3_long_bot_id =  0#long bot ALGO
    c3_short_bot_id = 0 #short bot
    c3_type = 'futures'
    c3_pyramiding = 3
    c3_max_deals = 1


    # iLen_sell = IntParameter(2, 15, default=7, space='sell', optimize=True)
    # iLZ_sell = DecimalParameter(0.01, 10.0, default=0.35, space='sell', optimize=True)
    # iELSTX_sell = IntParameter(5, 75, default=15, space='sell', optimize=True)
    # iGI_sell= DecimalParameter(0.01, 10.0, default=0.06, space='sell', optimize=True)

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
            'LMA': {'color': 'red'},
            'AP1': {'color': 'blue'},
            'AP2': {'color': 'yellow'},
            'G2': {'color': 'green'},
            'NEXTUP': {'color': 'white'},
            'NEXTDOWN': {'color': 'black'},
            'AP': {'color': 'green'}
        },
        'subplots': {
          
        }
    }




    # def refreshC3Trades(self):
    #     needRefreshC3 = False
    #     if not 'c3ActiveDeals' in self.custom_main:
    #         self.custom_main['c3ActiveDeals'] = 0

    #     if not 'c3LastRefresh' in self.custom_main:
    #         self.custom_main['c3LastRefresh'] = datetime.now(timezone.utc)
    #         needRefreshC3 = True
    #         logger.info(f"3Commas: first time refresh")
    #     elif (datetime.now(timezone.utc) - self.custom_main['c3LastRefresh']) > timedelta(minutes=30):
    #         self.custom_main['c3LastRefresh'] = datetime.now(timezone.utc)
    #         needRefreshC3 = True
    #         logger.info(f"3Commas: minutes time refresh")
    #     if needRefreshC3 == True:
    #         self.custom_main['c3ActiveDeals'] = 0
    #         p3cw = Py3CW(
    #                     key=self.c3_key,
    #                     secret=self.c3_secret,
    #                 )
    #         logger.info(f"3Commas: set refresh mode to {self.c3_mode}")
    #         error, data = p3cw.request(
    #                 entity='users',
    #                 action= 'change_mode',
    #                 payload={
    #                 "mode": f"{self.c3_mode}"
    #                 }
    #         )
    #         if error:
    #             logger.error(f"3Commas: mode {error['msg']}")
    #         else:
    #             logger.info(f"3Commas: mode {data}")
    #         error, data = p3cw.request(
    #                     entity='deals',
    #                     action= '',
    #                     payload={
    #                     "scope": "active" 
    #                     }
    #                 )
    #         if error:
    #             logger.error(f"3Commas: pairs {error['msg']}")
    #         else:
    #             logger.info(f"3Commas: get data")
    #             self.custom_3c_pairs = {}
    #             for deal in data:
    #                 p = deal["pair"]
    #                 logger.info(f"3Commas: pair {p}")
    #                 self.custom_3c_pairs[p] = {}
    #                 self.custom_3c_pairs[p]["LongId"] =  0
    #                 self.custom_3c_pairs[p]["ShortId"] = 0
    #                 if deal["bot_id"] == self.c3_long_bot_id:
    #                     self.custom_3c_pairs[p]["LongId"] = deal["id"]
    #                     logger.info(f"3Commas: pair {p} LongId{deal['id']}")
    #                     self.custom_main['c3ActiveDeals'] = self.custom_main['c3ActiveDeals'] + 1
    #                 if deal["bot_id"] == self.c3_short_bot_id:
    #                     self.custom_3c_pairs[p]["ShortId"] = deal["id"]
    #                     logger.info(f"3Commas: pair {p} ShortId{deal['id']}")
    #                     self.custom_main['c3ActiveDeals'] = self.custom_main['c3ActiveDeals'] + 1

    #         logger.info(f"3Commas: custom_3c_pairs {self.custom_3c_pairs}")


    def informative_pairs(self):

        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
    #    self.refreshC3Trades()
       grids = 6 
       Dir = 'neutral'
       MaIs = 'sma'
    #    if metadata['pair'] == 'EOS/USDT':
       self.iLen_buy.value = 7
       self.iLZ_buy.value = 0.09
       self.iELSTX_buy.value = 4
       self.iGI_buy.value = 0.45
       grids = 4
       Dir = 'down'
       MaIs = 'sma'

       if metadata['pair'] == 'LINK/USDT':
            self.iLen_buy.value = 7
            self.iLZ_buy.value = 3.5
            self.iELSTX_buy.value = 100000
            self.iGI_buy.value = 1
            grids = 4
            Dir = 'down'
            MaIs = 'sma'

       if metadata['pair'] == 'XLM/USDT':
            self.iLen_buy.value = 6
            self.iLZ_buy.value = 0.25
            self.iELSTX_buy.value = 0.5
            self.iGI_buy.value = 2.1
            grids = 4
            Dir = 'neutral'
            MaIs = 'sma'

       if metadata['pair'] == 'ADA/USDT':
            self.iLen_buy.value = 10
            self.iLZ_buy.value = 2.05
            self.iELSTX_buy.value = 17
            self.iGI_buy.value = 1.6
            grids = 6
            Dir = 'neutral'
            MaIs = 'ema'

       if metadata['pair'] == 'XMR/USDT':
            self.iLen_buy.value = 7
            self.iLZ_buy.value = 0.8
            self.iELSTX_buy.value = 0.04
            self.iGI_buy.value = 2
            grids = 4
            Dir = 'neutral'
            MaIs = 'lreg'

       funcGridBotAuto(dataframe, source = 'close', iLen = int(self.iLen_buy.value), iGrids = grids, iMA = MaIs, iLZ = self.iLZ_buy.value, iELSTX = self.iELSTX_buy.value, iGI = self.iGI_buy.value, iEXTR = False, iDir = Dir, iReset = True)
       dataframe[['lrc_up','lrc_down']] = funcLinearRegressionChannel2(dataframe, source = 'close', window = self.window_2_buy.value, deviations = self.deviations_buy.value)
       return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        
        # funcGridBotAuto(dataframe, source = 'close', iLen = int(self.iLen_buy.value), iGrids = 6, iMA = 'ema', iLZ = self.iLZ_buy.value, iELSTX = self.iELSTX_buy.value, iGI = self.iGI_buy.value, iEXTR = True, iDir = 'neutral', iReset = True)

        dataframe.loc[
            (
                ((dataframe['Buy_s'] == 1 ) |  (dataframe['Sell_s'] == 1 )) &
                # ((dataframe['Buy_s'] == 1 ) ) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # funcGridBotAuto(dataframe, source = 'close', iLen = int(self.iLen_sell.value), iGrids = 6, iMA = 'ema', iLZ = self.iLZ_sell.value, iELSTX = self.iELSTX_sell.value, iGI = self.iGI_sell.value, iEXTR = True, iDir = 'neutral', iReset = True)
        dataframe.loc[
            (
            #    (dataframe['Sell_s'] == 1) &
            # (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                last_candle = dataframe.iloc[-1].squeeze()
                previous_candle = dataframe.iloc[-2].squeeze()
                coin, currency = pair.split('/')

                if self.c3_type == 'spot':
                    c3_pair = f"{currency}_{coin}" #spot pair  
                else:
                    c3_pair = f"{currency}_{coin}{currency}" #future pairs binace

                p3cw = Py3CW(
                    key=self.c3_key,
                    secret=self.c3_secret,
                )
                c3_open_deals = {}
                c3_this_pair_open_count = 0
                #set mode real or paper
                logger.info(f"3Commas: set refresh mode to {self.c3_mode}")
                error, data = p3cw.request(
                        entity='users',
                        action= 'change_mode',
                        payload={
                        "mode": f"{self.c3_mode}"
                        }
                )
                if error:
                    logger.error(f"3Commas: mode {error['msg']}")
                else:
                    logger.info(f"3Commas: mode {data}")

                # get active deals
                error, data = p3cw.request(
                        entity='deals',
                        action= '',
                        payload={
                        "scope": "active" 
                        }
                    )
                if error:
                    logger.error(f"3Commas: pairs {error['msg']}")
                else:
                    logger.info(f"3Commas: get data")
                    self.custom_3c_pairs = {}
                    for deal in data:
                        open_pair = deal["pair"]
                        logger.info(f"3Commas: pair {open_pair}")
                        needCloseDeal = False
                        id_deal_close = deal["id"]
                        if deal["bot_id"] == self.c3_long_bot_id:
                           if open_pair ==  c3_pair:
                               if last_candle['Sell_s'] == 1:
                                   #close pair 
                                   needCloseDeal = True
                               else:
                                   c3_this_pair_open_count = c3_this_pair_open_count + 1
                           if needCloseDeal == False:
                                c3_open_deals[c3_pair] = True

                           
                        if deal["bot_id"] == self.c3_short_bot_id:
                           if open_pair ==  c3_pair:
                               if last_candle['Buy_s'] == 1:
                                   #close pair 
                                   needCloseDeal = True
                               else:
                                   c3_this_pair_open_count = c3_this_pair_open_count + 1
                           if needCloseDeal == False:
                                c3_open_deals[c3_pair] = True
                        
                        
                        if needCloseDeal == True:
                            logger.info(f"3Commas: Sending close signal for {open_pair} to 3commas bot_id={id_deal_close}")
                            errorCloseDeal, dataCloseDeal = p3cw.request(
                                entity='deals',
                                action='panic_sell',
                                action_id=f'{id_deal_close}' 
                            )

                            if errorCloseDeal:
                                logger.error(f"3Commas: {errorCloseDeal['msg']}")
                            else:
                                logger.info(f"3Commas: {dataCloseDeal}")

                    logger.info(f"3Commas: c3_this_pair_open_count= {c3_this_pair_open_count} c3_open_deals={len(c3_open_deals.keys())}")      
                    if c3_this_pair_open_count < self.c3_pyramiding:
                        bot_id = 0
                        type = ""
                        LRDir = True
                        if last_candle['Buy_s'] == 1:
                            bot_id = self.c3_long_bot_id #long bot ALGO
                            type = "buy"
                            # if last_candle['lrc_up'] > previous_candle['lrc_up']:
                            #    LRDir = True 
                        if last_candle['Sell_s'] == 1:
                            bot_id = self.c3_short_bot_id #short bot
                            type = "sell"
                            # if last_candle['lrc_up'] < previous_candle['lrc_up']:
                            #    LRDir = True
                        logger.info(f"3Commas: type= {type} bot_id={bot_id}")
                        if bot_id != 0 and LRDir == True:
                            logger.info(f"3Commas: Sending {type} signal for {c3_pair} to 3commas bot_id={bot_id}")
                            error, data = p3cw.request(
                                entity='bots',
                                action='start_new_deal',
                                action_id=f'{bot_id}',
                                payload={
                                    "bot_id": bot_id,
                                    "pair":  c3_pair,
                                    "skip_open_deals_checks": "true"
                                },
                            )

                            PairLocks.lock_pair(
                                pair=pair,
                                until=datetime.now(timezone.utc) + timedelta(minutes=5),
                                reason="3c lock pair"
                            )  
                

                
                return False
    
"""
Indicator factory
viktors.s@gmail.com
"""
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import math


def get(dtloc, indicatorType, source = 'close', length = 20, alpha = 0.7):

    if indicatorType == 'TMA':
        return ta.SMA((ta.SMA(dtloc[source], timeperiod = math.ceil(length / 2))), timeperiod= (math.floor(length / 2) + 1))
    elif indicatorType == 'MF':
        #Modular Filter
        return MF(dtloc, source, length)
    elif indicatorType == 'LSMA':
        return 0
    elif indicatorType == 'SMA': # Simple
        return ta.SMA(dtloc[source],timeperiod= length)
    elif indicatorType == 'EMA': # Exponential
        return ta.EMA(dtloc[source],timeperiod= length)
    elif indicatorType == 'DEMA': # Double Exponential
        return ta.DEMA(dtloc[source], timeperiod=length)
    elif indicatorType == 'TEMA': # Triple Exponential
        return ta.TEMA(dtloc[source], timeperiod=length)
    elif indicatorType == 'WMA':
        return ta.WMA(dtloc[source], timeperiod=length)
    elif indicatorType == 'VAMA':
        return 0
    elif indicatorType == 'HMA': # HULL
        return ta.WMA(
                    2 * ta.WMA(dtloc[source], int(math.floor(length/2))) - ta.WMA(dtloc[source], length), int(round(np.sqrt(length)))
                    )
    elif indicatorType == 'JMA': #Jurik
        return JMA(dtloc, source, length = length, jurik_phase = 3, jurik_power = 1)
    elif indicatorType == 'Kijun v2':
        return 0
    elif indicatorType == 'McGinley':
        return 0
    elif indicatorType == 'EDSMA': 
        #EDSMA - Super Smoother Filter 
        return EDSMA(dtloc, source, length = length, ssfPoles = 2, ssfLength = 20)
    elif indicatorType == 'LRSI':
        return LRSI(dtloc, source = source, alpha = alpha)
    elif indicatorType == 'TRANGE':
        #TRUE RANGE
        return ta.TRANGE(dtloc)
    elif indicatorType == 'KBC':
        #Keltner Baseline Channel
        return KBC(dtloc, source = source, length = length)



#zero lag EMA
def zlema(series, window=20):
    lag =  int(math.floor((window - 1) / 2) )
    ema_data = series + (series - series.shift(lag))
    return ta.EMA(ema_data, timeperiod = window)  


#zero lag HULL
def zlhull(series, window=20):
    lag =  int(math.floor((window - 1) / 2) )
    wma_data = series + (series - series.shift(lag))
    return  ta.WMA(
                    2 * ta.WMA(wma_data, int(math.floor(window/2))) - ta.WMA(wma_data, window), int(round(np.sqrt(window)))
                    )


#Keltner Baseline Channel Color
def KBC(dtloc, source = 'close', indicatorType = 'HMA', length = 20, useTrueRange = True, multy = 0.2 ):
    #multy - Base Channel Multiplier

    dfKBC = dtloc#.copy()
    dfKBC['BBMC'] = get(dfKBC, indicatorType = indicatorType, source = 'close', length = length)
    dfKBC['Keltma'] = get(dfKBC, indicatorType = indicatorType, source = source, length = length)
    if useTrueRange == True:
        dfKBC['KBCrange'] = get(dfKBC, indicatorType = 'TRANGE')
    else:
        dfKBC['KBCrange']  = dfKBC['high'] - dfKBC['low']
    dfKBC['KBCrangeEma']  = get(dfKBC, indicatorType = 'EMA', source = 'KBCrange', length = length)

    dfKBC['KBC_upperk']  =  dfKBC['Keltma']  + dfKBC['KBCrangeEma'] * multy
    dfKBC['KBC_lowerk']  =  dfKBC['Keltma']  - dfKBC['KBCrangeEma'] * multy

    dfKBC['KBCColor']  =  np.where(dfKBC['close'] > dfKBC['KBC_upperk'],'blue',np.where(dfKBC['close'] < dfKBC['KBC_lowerk'],'red', 'gray'))
    return dfKBC['KBCColor']


# def McGinley(dtloc, source = 'close', length = 20):
#     dtM = dtloc.copy()
#     def calc_McGinley(dfr):
#         global calc_McGinley_mg
#         try:
#             calc_McGinley_mg
#         except NameError:
#             calc_McGinley_mg = 0
#         if math.isnan(calc_McGinley_mg) == True:
#             calc_McGinley_mg = 0
#         if calc_McGinley_mg > 
        

   
# pine_rma(src, length) =>
# 	alpha = 1/length
# 	sum = 0.0
# 	sum := na(sum[1]) ? sma(src, length) : alpha * src + (1 - alpha) * nz(sum[1])
def RMA(dtloc, source = 'close', length = 20):
    dtRMA = dtloc.copy()
    dtRMA['RMASMATMP'] = ta.SMA(dtloc[source],timeperiod= length)
    calc_RMA_value = 0.0
    def calc_RMA(dfr):
        global calc_RMA_value
        try:
            calc_RMA_value
        except NameError:
            calc_RMA_value = 0.0
        if math.isnan(calc_RMA_value) == True:
            calc_RMA_value = 0.0
        alpha = 1/length
        if calc_RMA_value == 0:
            calc_RMA_value = dfr['RMASMATMP']
        else:
            calc_RMA_value = alpha * dfr[source] + (1- alpha) * calc_RMA_value
        return calc_RMA_value
    dtRMA['RMA'] = dtRMA.calc_RMA(calc_RMA, axis = 1)
    return dtRMA['RMA'] 



def TDSeq(dtloc, source = 'close'):
    dtTDSeq = dtloc.copy()
    # calc_dtTDSeq_source = list()
    # calc_dtTDSeq_buySetup = 0
    retcol = 'TDSeqcol'
    def calc_dtTDSeq(dfr, init = 0):
        global calc_dtTDSeq_source
        global calc_dtTDSeq_buySetup
        if init == 1:
            calc_dtTDSeq_source = list()
            calc_dtTDSeq_buySetup = 0 
            return
        # try:
        #     calc_dtTDSeq_source
        # except NameError:
        #     calc_dtTDSeq_source = list()
      
        # try:
        #     calc_dtTDSeq_buySetup
        # except NameError:
        #     calc_dtTDSeq_buySetup = 0
        
        calc_dtTDSeq_source.append(dfr[source])
        if len(calc_dtTDSeq_source) >= 5:
             if calc_dtTDSeq_source[-1] > calc_dtTDSeq_source[-5] or 0 < calc_dtTDSeq_buySetup:
                 calc_dtTDSeq_buySetup = calc_dtTDSeq_buySetup + 1
             else:
                 calc_dtTDSeq_buySetup = 0
        else:
            calc_dtTDSeq_buySetup = 0
        
        return calc_dtTDSeq_buySetup
    calc_dtTDSeq(None, init=1)
    dtTDSeq[retcol] = dtTDSeq.apply(calc_dtTDSeq, axis = 1)
    return dtTDSeq[retcol]



                 













def RSIDivergence(dtloc, source = 'close', source2= 'rsi',  length = 14, barrier = 30, without_barrier = False, reverse = False):

     dtRSIDiv = dtloc.copy().fillna(0)
     locrsicol =  source2
     retcol = 'RSIDivergence'
     dummy = 99999999
    #  calc_RSIDivergence_source = list()
    #  calc_RSIDivergence_rsi = list()
     def calc_RSIDivergence(dfr, init = 0):
        global calc_RSIDivergence_source
        global calc_RSIDivergence_rsi
        if init == 1:
            calc_RSIDivergence_source = list()
            calc_RSIDivergence_rsi = list()
            return
        is_divergence = 0
        if len(calc_RSIDivergence_source) >= length:
            min_rsi_value = dummy
            min_rsi_price = dummy
            min_index = dummy
            b_is_above = False
            for val in reversed(range(length-1)):
                if calc_RSIDivergence_rsi[val] < min_rsi_value:
                   min_index = val
                   min_rsi_value =  calc_RSIDivergence_rsi[val]
                   min_rsi_price =  calc_RSIDivergence_source[val]
            for val in reversed(range(min_index, length-1)):
                if (calc_RSIDivergence_rsi[val] > barrier and without_barrier == False) or (calc_RSIDivergence_rsi[val] > min_rsi_value and  without_barrier == True):
                    b_is_above = True
            if reverse == False and  b_is_above == True and ((min_rsi_value < barrier and dfr[locrsicol] < barrier) or without_barrier == True) and  min_rsi_value < dfr[locrsicol]  and min_rsi_price > dfr[source]:
                is_divergence = 1
            if reverse == True and  b_is_above == True and ((min_rsi_value < barrier and dfr[locrsicol] < barrier) or without_barrier == True) and  min_rsi_value > dfr[locrsicol]  and min_rsi_price < dfr[source]:
                is_divergence = 1
            calc_RSIDivergence_source.pop(0)
            calc_RSIDivergence_rsi.pop(0)
        calc_RSIDivergence_source.append(dfr[source])
        calc_RSIDivergence_rsi.append(dfr[locrsicol])
        return is_divergence
     calc_RSIDivergence(None, init=1)
     dtRSIDiv[retcol] = dtRSIDiv.apply(calc_RSIDivergence, axis = 1)
     return dtRSIDiv[retcol]


    
#Kalman Filter Filter
def KalmanFilter(dtloc, source = 'close'):
    
    dtKF = dtloc.copy().fillna(0)
    dtKF['TRANGE'] = ta.TRANGE(dtloc).fillna(0)


    def calc_dtKF(dfr, init=0):
        global calc_dtKF_value_1
        global calc_dtKF_value_2
        global calc_dtKF_value_3
        global calc_dtKF_source
        if init == 1:
            calc_dtKF_value_1 = 0.0
            calc_dtKF_value_2 = 0.0
            calc_dtKF_value_3 = 0.0
            calc_dtKF_source = 0.0
            return
        calc_dtKF_value_1 = 0.2 * (dfr[source] - calc_dtKF_source) + 0.8 * calc_dtKF_value_1
        calc_dtKF_value_2 = 0.1 * dfr['TRANGE'] + 0.8 * calc_dtKF_value_2
        if calc_dtKF_value_2 != 0:
            vlambda = abs(calc_dtKF_value_1/calc_dtKF_value_2)
        else:
            vlambda = 0
        valpha =  (-1*math.pow(vlambda,2) + math.sqrt(math.pow(vlambda,4) + 16 * math.pow(vlambda,2)))/8
        calc_dtKF_value_3 = valpha * dfr[source] + (1 - valpha) * calc_dtKF_value_3
        calc_dtKF_source = dfr[source]
       
        return calc_dtKF_value_3
    calc_dtKF(None, init=1)
    dtKF['KF'] = dtKF.apply(calc_dtKF, axis = 1)
    return dtKF['KF'] 

# #Kalman Filter Filter
# def KalmanFilter(dtloc, source = 'close'):
    
#     dtKF = dtloc.copy()
#     dtKF['TRANGE'] = ta.TRANGE(dtloc)
#     calc_dtKF_value_1 = 0.0
#     calc_dtKF_value_2 = 0.0
#     calc_dtKF_value_3 = 0.0
#     calc_dtKF_source = 0.0

#     def calc_dtKF(dfr):
#         global calc_dtKF_value_1
#         try:
#             calc_dtKF_value_1
#         except NameError:
#             calc_dtKF_value_1 = 0.0
#         if math.isnan(calc_dtKF_value_1) == True:
#             calc_dtKF_value_1 = 0.0
        
#         global calc_dtKF_value_2
#         try:
#             calc_dtKF_value_2
#         except NameError:
#             calc_dtKF_value_2 = 0.0
#         if math.isnan(calc_dtKF_value_2) == True:
#             calc_dtKF_value_2 = 0.0
        
#         global calc_dtKF_value_3
#         try:
#             calc_dtKF_value_3
#         except NameError:
#             calc_dtKF_value_3 = 0.0
#         if math.isnan(calc_dtKF_value_3) == True:
#             calc_dtKF_value_3 = 0.0
        
#         global calc_dtKF_source
#         try:
#             calc_dtKF_source
#         except NameError:
#             calc_dtKF_source = 0.0
#         if math.isnan(calc_dtKF_source) == True:
#             calc_dtKF_source = 0.0

#         calc_dtKF_value_1 = 0.2 * (dfr[source] - calc_dtKF_source) + 0.8 * calc_dtKF_value_1
#         calc_dtKF_value_2 = 0.1 * dfr['TRANGE'] + 0.8 * calc_dtKF_value_2
#         if calc_dtKF_value_2 != 0:
#             vlambda = abs(calc_dtKF_value_1/calc_dtKF_value_2)
#         else:
#             vlambda = 0
#         valpha =  (-1*math.pow(vlambda,2) + math.sqrt(math.pow(vlambda,4) + 16 * math.pow(vlambda,2)))/8
#         calc_dtKF_value_3 = valpha * dfr[source] + (1 - valpha) * calc_dtKF_value_3
#         calc_dtKF_source = dfr[source]
#         return calc_dtKF_value_3
#     dtKF['KF'] = dtKF.apply(calc_dtKF, axis = 1)
#     return dtKF['KF'] 


#Differences of Exponentially Weighted Averages
def DiffEWA(dataframe, source = 'close', beta = 0.01, length = 1):
    df = dataframe.copy().fillna(0)
    def calc_ewa(dfr, init=0):
        global calc_ewa_value
        global calc_src_value
        if init == 1:
            calc_ewa_value = [0.0] * length
            calc_src_value = [0.0] * length
            return
        calc_src_value.pop(0)
        calc_src_value.append(dfr[source])
        val = beta * calc_src_value[0] + (1-beta) * calc_ewa_value[-1]
        calc_ewa_value.pop(0)
        calc_ewa_value.append(val)
        return val
    calc_ewa(None, init=1)
    df['ewa'] = df.apply(calc_ewa, axis = 1)
    df['dewa'] = beta * (df['ewa'] - df['ewa'].shift(1))  + (1-beta) * (df['ewa'] - df['ewa'].shift(1))
    
    return df['ewa'], df['dewa']




def LUX_SuperTrendOscillator(dtloc, source = 'close', length = 6, mult = 9, smooth = 72):
    """
      https://www.tradingview.com/script/dVau7zqn-LUX-SuperTrend-Oscillator/
     :return: List of tuples in the format (osc, ama, hist)   
    """
    def_proc_name = '_LUX_SuperTrendOscillator'
    atrcol        = 'atr'    + def_proc_name
    hl2col        = 'hl2'    + def_proc_name
    upcol         = 'up'     + def_proc_name
    dncol         = 'dn'     + def_proc_name
    uppercol      = 'upper'  + def_proc_name
    lowercol      = 'lower'  + def_proc_name
    trendcol      = 'trend'  + def_proc_name
    sptcol        = 'spt'    + def_proc_name
    osc1col       = 'osc1'   + def_proc_name
    osc2col       = 'osc2'   + def_proc_name
    osccol        = 'osc'    + def_proc_name
    alphacol      = 'alpha'  + def_proc_name
    amacol        = 'ama'    + def_proc_name
    histcol       = 'hist'   + def_proc_name


    dtS = dtloc.copy().fillna(0)
    dtS[atrcol] = ta.ATR(dtloc, timeperiod = length) * mult
    dtS[hl2col] =  (dtS['high'] + dtS['low'] )/2
    dtS[upcol] =  dtS[hl2col] + dtS[atrcol]
    dtS[dncol] =  dtS[hl2col] - dtS[atrcol]
    def calc_upper(dfr, init=0):
        global calc_Lux_STO_upper
        global calc_Lux_STO_src
        if init == 1:
            calc_Lux_STO_upper = 0.0
            calc_Lux_STO_src = 0.0
            return
        if calc_Lux_STO_src < calc_Lux_STO_upper:
            calc_Lux_STO_upper = min(dfr[upcol], calc_Lux_STO_upper)
        else:
            calc_Lux_STO_upper = dfr[upcol]
        calc_Lux_STO_src = dfr[source]
        return calc_Lux_STO_upper
    calc_upper(None, init=1)
    dtS[uppercol] = dtS.apply(calc_upper, axis = 1)
    def calc_lower(dfr, init=0):
        global calc_Lux_STO_lower
        global calc_Lux_STO_src
        if init == 1:
            calc_Lux_STO_lower = 0.0
            calc_Lux_STO_src = 0.0
            return
        if calc_Lux_STO_src > calc_Lux_STO_lower:
            calc_Lux_STO_lower= max(dfr[dncol], calc_Lux_STO_lower)
        else:
            calc_Lux_STO_lower = dfr[dncol]
        calc_Lux_STO_src = dfr[source]
        return calc_Lux_STO_lower
    calc_lower(None, init=1)
    dtS[lowercol] = dtS.apply(calc_lower, axis = 1)
    def calc_trend(dfr, init=0):
        global calc_Lux_STO_trend
        global calc_Lux_STO_lower
        global calc_Lux_STO_upper
        if init == 1:
            calc_Lux_STO_trend = 0.0
            calc_Lux_STO_lower = 0.0
            calc_Lux_STO_upper = 0.0
            return
        if dfr[source] > calc_Lux_STO_upper:
            calc_Lux_STO_trend = 1
        elif dfr[source] < calc_Lux_STO_lower:
            calc_Lux_STO_trend = 0
        calc_Lux_STO_upper = dfr[uppercol]
        calc_Lux_STO_lower = dfr[lowercol]
        return calc_Lux_STO_trend
    calc_trend(None, init=1)
    dtS[trendcol] = dtS.apply(calc_trend, axis = 1)
    dtS[sptcol] = dtS[trendcol] * dtS[lowercol] + (1-dtS[trendcol] ) * dtS[uppercol]
    dtS[osc1col] = (dtS[source] - dtS[sptcol]) / (dtS[uppercol] - dtS[lowercol])
    dtS[osc2col] = np.where(dtS[osc1col] < 1, dtS[osc1col], 1 )
    dtS[osccol] = np.where(dtS[osc2col] > -1, dtS[osc2col], -1)
    dtS[alphacol] = dtS[osccol].pow(2)/length
    def calc_ama(dfr, init=0):
        global calc_Lux_STO_ama
        if init == 1:
            calc_Lux_STO_ama = 0.0
            return
        calc_Lux_STO_ama = calc_Lux_STO_ama + dfr[alphacol] * (dfr[osccol] - calc_Lux_STO_ama)
        return calc_Lux_STO_ama
    calc_ama(None, init=1)
    dtS[amacol] = dtS.apply(calc_ama, axis = 1)
    dtS[histcol] = ta.EMA((dtS[osccol]- dtS[amacol]),timeperiod = smooth)

    return dtS[osccol] * 100,  dtS[amacol] * 100 , dtS[histcol]  * 100, dtS[sptcol] 
    




def Z(dtloc, source = 'close', length = 20):
    dtZ = dtloc.copy().fillna(0)
    colret = 'ZColName'
    def calc_Z(dfr, init=0):
        global calc_Z_source
        if init == 1:
            calc_Z_source = list()
            return
        calc_Z_source.append(dfr[source])    
        h = 0.0
        d = 0.0
        for i in range(len(calc_Z_source)):
            k = (length - i) * length
            h = h + k
            d = d + calc_Z_source[i] * k
        if len(calc_Z_source) > length:
            calc_Z_source.pop(0)
        return d/h
    calc_Z(None, init=1)
    dtZ[colret] = dtZ.apply(calc_Z, axis = 1)
    return dtZ[colret]     







#Modular Filter
def MFv2(dtloc, source = 'close', length = 20, feedback = False, feedback_weight = 0.5, feedback_filter = 0.8):
    
    dtM = dtloc.copy().fillna(0)

    
    def calc_MF_ts(dfr, init=0):
        global calc_MF_ts_ts
        global calc_MF_ts_os
        global calc_MF_ts_b
        global calc_MF_ts_c

        if init == 1:
            calc_MF_ts_ts = 0.0
            calc_MF_ts_os = 0.0
            calc_MF_ts_b =  0.0
            calc_MF_ts_c =  0.0
            return

        
        # try:
        #     calc_MF_ts_ts
        # except NameError:
        #     calc_MF_ts_ts = 0.0
        # if math.isnan(calc_MF_ts_ts) == True:
        #     calc_MF_ts_ts = dfr[source]
        # global calc_MF_ts_os
        # try:
        #     calc_MF_ts_os
        # except NameError:
        #     calc_MF_ts_os = 0
        # if math.isnan(calc_MF_ts_os) == True:
        #     calc_MF_ts_os = 0
        # global calc_MF_ts_b
        # try:
        #     calc_MF_ts_b
        # except NameError:
        #     calc_MF_ts_b = 0.0
        
        # global calc_MF_ts_c
        # try:
        #     calc_MF_ts_c
        # except NameError:
        #     calc_MF_ts_c = 0.0
        
        alpha = 2/(length+1)

        if feedback == True:
            a = dfr[source] * feedback_weight + (1-feedback_weight) * calc_MF_ts_ts
        else:
            a = dfr[source]
        
        if math.isnan(calc_MF_ts_b) == True:
            calc_MF_ts_b = a
        if math.isnan(calc_MF_ts_c) == True:
            calc_MF_ts_c = a

        if a > (alpha * a + (1-alpha) * calc_MF_ts_b):
            calc_MF_ts_b = a
        else:
            calc_MF_ts_b = (alpha * a + (1-alpha) * calc_MF_ts_b)
        
        if a < (alpha * a + (1-alpha) * calc_MF_ts_c):
            calc_MF_ts_c = a
        else:
            calc_MF_ts_c = (alpha * a + (1-alpha) * calc_MF_ts_c)

        if ( a == calc_MF_ts_b):
            calc_MF_ts_os = 1
        elif (a == calc_MF_ts_c):
            calc_MF_ts_os = 0

        upper = feedback_filter * calc_MF_ts_b + (1-feedback_filter) * calc_MF_ts_c
        lower = feedback_filter * calc_MF_ts_c + (1-feedback_filter) * calc_MF_ts_b 
        calc_MF_ts_ts  = calc_MF_ts_os * upper + (1-calc_MF_ts_os) * lower
        return  calc_MF_ts_os #pd.Series([calc_MF_ts_ts,])
    
    calc_MF_ts(None, init=1)
    dtM['MF']  = dtM.apply(calc_MF_ts, axis = 1)

    return dtM['MF']





def MF(dtloc, source = 'close', length = 20, feedback = False, feedback_weight = 0.5, feedback_filter = 0.8):
    
    dtM = dtloc.copy()

    calc_MF_ts_ts = 0
    calc_MF_ts_os = 0
    calc_MF_ts_b = 0
    calc_MF_ts_c = 0
    def calc_MF_ts(dfr):
        global calc_MF_ts_ts
        try:
            calc_MF_ts_ts
        except NameError:
            calc_MF_ts_ts = 0.0
        if math.isnan(calc_MF_ts_ts) == True:
            calc_MF_ts_ts = dfr[source]
        global calc_MF_ts_os
        try:
            calc_MF_ts_os
        except NameError:
            calc_MF_ts_os = 0
        if math.isnan(calc_MF_ts_os) == True:
            calc_MF_ts_os = 0
        global calc_MF_ts_b
        try:
            calc_MF_ts_b
        except NameError:
            calc_MF_ts_b = 0.0
        
        global calc_MF_ts_c
        try:
            calc_MF_ts_c
        except NameError:
            calc_MF_ts_c = 0.0
        
        alpha = 2/(length+1)

        if feedback == True:
            a = dfr[source] * feedback_weight + (1-feedback_weight) * calc_MF_ts_ts
        else:
            a = dfr[source]
        
        if math.isnan(calc_MF_ts_b) == True:
            calc_MF_ts_b = a
        if math.isnan(calc_MF_ts_c) == True:
            calc_MF_ts_c = a

        if a > (alpha * a + (1-alpha) * calc_MF_ts_b):
            calc_MF_ts_b = a
        else:
            calc_MF_ts_b = (alpha * a + (1-alpha) * calc_MF_ts_b)
        
        if a < (alpha * a + (1-alpha) * calc_MF_ts_c):
            calc_MF_ts_c = a
        else:
            calc_MF_ts_c = (alpha * a + (1-alpha) * calc_MF_ts_c)

        if ( a == calc_MF_ts_b):
            calc_MF_ts_os = 1
        elif (a == calc_MF_ts_c):
            calc_MF_ts_os = 0

        upper = feedback_filter * calc_MF_ts_b + (1-feedback_filter) * calc_MF_ts_c
        lower = feedback_filter * calc_MF_ts_c + (1-feedback_filter) * calc_MF_ts_b 
        calc_MF_ts_ts  = calc_MF_ts_os * upper + (1-calc_MF_ts_os) * lower
        return calc_MF_ts_ts
    dtM['MF'] = dtM.apply(calc_MF_ts, axis = 1)

    return dtM['MF'] 

def JMA(dtloc, source = 'close', length = 5, jurik_phase = 3, jurik_power = 1):
    dfj = dtloc.copy()
    if jurik_phase < -100:
        phaseRatio = 0.5
    elif jurik_phase > 100:
        phaseRatio = 2.5
    else:
        phaseRatio = jurik_phase / 100 + 1.5
    
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = math.pow(beta, jurik_power)
    calc_e0_e0 = 0
    def calc_e0(dfr):
        global calc_e0_e0
        try:
            calc_e0_e0
        except NameError:
            calc_e0_e0 = 0.0
        if math.isnan(calc_e0_e0) == True:
            calc_e0_e0 = 0.0
        calc_e0_e0 = (1 - alpha) * dfr[source] + alpha * calc_e0_e0
        return calc_e0_e0
    dfj['e0'] = dfj.apply(calc_e0, axis = 1)

    calc_e1_e1 = 0
    def calc_e1(dfr):
        global calc_e1_e1
        try:
            calc_e1_e1
        except NameError:
            calc_e1_e1 = 0.0
        if math.isnan(calc_e1_e1) == True:
            calc_e1_e1= 0.0
        #e1 := (src - e0) * (1 - beta) + beta * nz(e1[1])
        calc_e1_e1 = (dfr[source] - dfr['e0']) * (1 - beta) + beta * calc_e1_e1
        return calc_e1_e1
    dfj['e1'] = dfj.apply(calc_e1, axis = 1)

    calc_e2_e2 = 0
    calc_e2_jma = 0
    def calc_e2(dfr):
        global calc_e2_e2
        try:
            calc_e2_e2
        except NameError:
            calc_e2_e2 = 0.0
        if math.isnan(calc_e2_e2) == True:
            calc_e2_e2= 0.0

        global calc_e2_jma
        try:
            calc_e2_jma
        except NameError:
            calc_e2_jma = 0.0
        if math.isnan(calc_e2_jma) == True:
            calc_e2_jma= 0.0

        # e2 := (e0 + phaseRatio * e1 - nz(jma[1])) * pow(1 - alpha, 2) + pow(alpha, 2) * nz(e2[1])
        calc_e2_e2 = (dfr['e0'] + phaseRatio * dfr['e1'] - calc_e2_jma) * math.pow(1 - alpha, 2) + math.pow(alpha, 2) * calc_e2_e2 
        calc_e2_jma = calc_e2_e2 + calc_e2_jma # fucking future :D
        return calc_e2_e2
    dfj['e2'] = dfj.apply(calc_e2, axis = 1)

    calc_jma = 0
    def calc_jma(dfr):
        global calc_jma
        try:
            calc_jma
        except NameError:
            calc_jma = 0.0
        if math.isnan(calc_jma) == True:
            calc_jma = 0.0
        # jma := e2 + nz(jma[1])
        calc_jma = dfr['e2'] +  calc_jma 
        return calc_jma
    dfj['jma'] = dfj.apply(calc_jma, axis = 1)
    return dfj['jma']






def LRSI(dtloc, source = 'close', alpha = 0.7):
    dfL = dtloc.copy()
    alphaL = alpha
    # L0 = 0.0
    # L0 := alpha * src + (1 - alpha) * nz(L0[1])
    l0 = 0
    def calc_l0(df):
        global l0
        try:
            l0
        except NameError:
            l0 = 0.0
        if math.isnan(l0) == True:
            l0 = 0.0
        l0 = alphaL * df[source] + ( 1 - alphaL ) * l0
        return l0
    dfL['L0'] = dfL.apply(calc_l0, axis = 1)


    # L1 = 0.0
    # L1 := -(1 - alpha) * L0 + nz(L0[1]) + (1 - alpha) * nz(L1[1])
    l1 = 0
    l0_1 = 0
    def calc_l1(df):
        global l1
        global l0_1
        try:
            l1
        except NameError:
            l1 = 0.0
        if math.isnan(l1) == True:
            l1 = 0.0
        try:
            l0_1
        except NameError:
            l0_1 = 0.0
        if math.isnan(l0_1) == True:
            l0_1 = 0.0
        l1 = (-(1 - alphaL)) * df['L0'] + l0_1 + (1 - alphaL) * l1
        l0_1 = df['L0']
        return l1
    dfL['L1'] = dfL.apply(calc_l1, axis = 1)
       
    # L2 = 0.0
    # L2 := -(1 - alpha) * L1 + nz(L1[1]) + (1 - alpha) * nz(L2[1])
    l2 = 0
    l1_1 = 0
    def calc_l2(df):
        global l2
        global l1_1
        try:
            l2
        except NameError:
            l2 = 0.0
        if math.isnan(l2) == True:
            l2 = 0.0
        try:
            l1_1
        except NameError:
            l1_1 = 0.0
        if math.isnan(l1_1) == True:
            l1_1 = 0.0
        l2 = (-(1 - alphaL)) * df['L1'] + l1_1 + (1 - alphaL) * l2
        l1_1 = df['L1']
        return l2
    dfL['L2'] =  dfL.apply(calc_l2, axis = 1)
       
        
    # L3 = 0.0
    # L3 := -(1 - alpha) * L2 + nz(L2[1]) + (1 - alpha) * nz(L3[1])
    l3 = 0
    l2_1 = 0
    def calc_l3(df):
        global l3
        global l2_1
        try:
            l3
        except NameError:
            l3 = 0.0
        if math.isnan(l3) == True:
            l3 = 0.0
        try:
            l2_1
        except NameError:
            l2_1 = 0.0
        if math.isnan(l2_1) == True:
            l2_1 = 0.0
        l3 = (-(1 - alphaL)) * df['L2'] + l2_1 + (1 - alphaL) * l3
        l2_1 = df['L2']
        return l3
    dfL['L3'] = dfL.apply(calc_l3, axis = 1)

        
    # CU = 0.0
    # CU := (L0 >= L1 ? L0 - L1 : 0) + (L1 >= L2 ? L1 - L2 : 0) + (L2 >= L3 ? L2 - L3 : 0)
    def calc_cu(df):
        cu1 = 0.0
        if (df['L0'] >= df['L1']):
            cu1 = df['L0'] - df['L1']
        cu2 = 0.0   
        if (df['L1'] >= df['L2']):
            cu2 = df['L1'] - df['L2']
        cu3 = 0.0 
        if (df['L2'] >= df['L3']):
            cu3 = df['L2'] - df['L3']
        return (cu1+cu2+cu3)
    dfL['CU'] = dfL.apply(calc_cu, axis = 1)

    # CD = 0.0
    # CD := (L0 >= L1 ? 0 : L1 - L0) + (L1 >= L2 ? 0 : L2 - L1) + (L2 >= L3 ? 0 : L3 - L2)
    def calc_dd(df):
        cd1 = 0.0
        if (df['L0'] >= df['L1']):
            cd1 = 0.0
        else:
             cd1 = df['L1'] - df['L0']
        cd2 = 0.0   
        if (df['L1'] >= df['L2']):
            cd2 = 0.0 
        else:
            cd2 = df['L2'] - df['L1']
        cd3 = 0.0 
        if (df['L2'] >= df['L3']):
            cd3 = 0.0 
        else:
            cd3 = df['L3'] - df['L2']
        return (cd1+cd2+cd3)
    dfL['CD'] = dfL.apply(calc_dd, axis = 1)
    def calc_lrsi(df):
        lrsi = 0.0
        if ((df['CU'] + df['CD']) != 0):
            lrsi = (df['CU']/(df['CU'] + df['CD']))
        return lrsi
    dfL['LRSI'] = dfL.apply(calc_lrsi, axis = 1)
    return dfL['LRSI']




#EDSMA - Super Smoother Filter 
def EDSMA(dtloc, source = 'close', length = 40, ssfPoles = 2, ssfLength = 20):
    
    df = dtloc.copy()
    get2PoleSSF_ssf_1 = 0
    get2PoleSSF_ssf_2 = 0
    def get2PoleSSF(dfr):
        global get2PoleSSF_ssf_1
        global get2PoleSSF_ssf_2

        try:
            get2PoleSSF_ssf_1
        except NameError:
            get2PoleSSF_ssf_1 = 0.0
        if math.isnan(get2PoleSSF_ssf_1) == True:
            get2PoleSSF_ssf_1 = 0.0
        try:
            get2PoleSSF_ssf_2
        except NameError:
            get2PoleSSF_ssf_2 = 0.0
        if math.isnan(get2PoleSSF_ssf_2) == True:
            get2PoleSSF_ssf_2 = 0.0

        arg = math.sqrt(2) * math.pi/ssfLength
        a1 = math.exp(-arg)
        b1 = 2 * a1 * math.cos(arg)
        c2 = b1
        c3 = -math.pow(a1, 2)
        c1 = 1 - c2 - c3
        src = dfr['avgZeros']
        ssf = c1 * src + c2 * get2PoleSSF_ssf_1 + c3 * get2PoleSSF_ssf_2
        #print(f'c1 = {c1} src = {src} c2 = {c2} get2PoleSSF_ssf_1 = {get2PoleSSF_ssf_1} c3 = {c3} get2PoleSSF_ssf_2 = {get2PoleSSF_ssf_2} ')
        get2PoleSSF_ssf_2 = get2PoleSSF_ssf_1
        get2PoleSSF_ssf_1 = ssf
        #print(f'ssf = {ssf}')
        return ssf
    get3PoleSSF_ssf_1 = 0
    get3PoleSSF_ssf_2 = 0
    get3PoleSSF_ssf_3 = 0
    def get3PoleSSF(dfr):
        global get3PoleSSF_ssf_1
        global get3PoleSSF_ssf_2
        global get3PoleSSF_ssf_3
        try:
            get3PoleSSF_ssf_1
        except NameError:
            get3PoleSSF_ssf_1 = 0.0
        if math.isnan(get3PoleSSF_ssf_1) == True:
            get3PoleSSF_ssf_1 = 0.0
        try:
            get3PoleSSF_ssf_2
        except NameError:
            get3PoleSSF_ssf_2 = 0.0
        if math.isnan(get3PoleSSF_ssf_2) == True:
            get3PoleSSF_ssf_2 = 0.0
        try:
            get3PoleSSF_ssf_3
        except NameError:
            get3PoleSSF_ssf_3 = 0.0
        if math.isnan(get3PoleSSF_ssf_3) == True:
            get3PoleSSF_ssf_3 = 0.0

        arg =  math.pi/ssfLength
        a1 = math.exp(-arg)
        b1 = 2 * a1 * math.cos(1.738 * arg)
        c1 = math.pow(a1,2)

        coef2 = b1 + c1
        coef3 = -(c1 + b1 * c1)
        coef4 = math.pow(c1, 2)
        coef1 = 1 - coef2 - coef3 - coef4
        ssf = coef1 * dfr['avgZeros'] + coef2 * get3PoleSSF_ssf_1 + coef3 * get3PoleSSF_ssf_2 + coef4 * get3PoleSSF_ssf_3
        get3PoleSSF_ssf_3 = get3PoleSSF_ssf_2
        get3PoleSSF_ssf_2 = get3PoleSSF_ssf_1
        get3PoleSSF_ssf_1 = ssf
        return ssf

    df['zeros'] = df[source] - df[source].shift(2)
    df['avgZeros']  = (df['zeros'] + df['zeros'].shift(1))/2
    if ssfPoles == 2:
        df['ssf'] = df.apply(get2PoleSSF, axis = 1)
    else:
        df['ssf'] = df.apply(get3PoleSSF, axis = 1)

    df['stdev'] = df['ssf'].rolling(length).std()

    def calc_scaledFilter(dfr):
        if dfr['stdev'] != 0:
            return (dfr['ssf']/dfr['stdev'])
        else:
            return 0


    df['scaledFilter'] = df.apply(calc_scaledFilter, axis = 1)

    def calc_alpha(dfr):
        return 5 * abs(dfr['scaledFilter'])/length

    df['alpha'] = df.apply(calc_alpha, axis = 1)

    calc_edsma_edsma = 0
    def calc_edsma(dfr):
        global calc_edsma_edsma
        try:
            calc_edsma_edsma
        except NameError:
            calc_edsma_edsma = 0.0
        if math.isnan(calc_edsma_edsma) == True:
            calc_edsma_edsma = 0.0
            
        alf = dfr['alpha']
        cl = dfr[source] 
        edsma = alf * cl + (1 - alf) * calc_edsma_edsma
       # print(f'alpha ={alf} close ={cl}  edsma ={edsma} calc_edsma_edsma ={calc_edsma_edsma}')
        calc_edsma_edsma = edsma
        return calc_edsma_edsma
    
    df['EDSMA'] = df.apply(calc_edsma, axis = 1)

    return df['EDSMA']

def getSSLChannel(dtloc, indicatorType = 'HMA', length = 60, exitBuySell = False):
    
    dfSSL = dtloc.copy()
    dfSSL['dfSSL_high']  = get(dfSSL, indicatorType = indicatorType, source = 'high', length = length)
    dfSSL['dfSSL_low']   = get(dfSSL, indicatorType = indicatorType, source = 'low', length = length)

    if exitBuySell == True:
        calc_hlv3_hlv3 = 0
        def calc_hlv3(dfr):
            global calc_hlv3_hlv3
            try:
                calc_hlv3_hlv3
            except NameError:
                calc_hlv3_hlv3 = 1
            if math.isnan(calc_hlv3_hlv3) == True:
                calc_hlv3_hlv3 = 1
            if dfr['close'] >  dfr['dfSSL_high']:
                calc_hlv3_hlv3 = 1
            elif dfr['close'] <  dfr['dfSSL_low']:
                calc_hlv3_hlv3 = -1
            return calc_hlv3_hlv3
        dfSSL['dfSSL_hlv3'] = dfSSL.apply(calc_hlv3, axis = 1)
        dfSSL['dfSSL_Exit'] = np.where(dfSSL['dfSSL_hlv3'] < 0,dfSSL['dfSSL_high'],dfSSL['dfSSL_low'])

        return dfSSL['dfSSL_Exit'], dfSSL['dfSSL_high'], dfSSL['dfSSL_low'], dfSSL['dfSSL_hlv3']

    else:
        return dfSSL['dfSSL_high'], dfSSL['dfSSL_low']

#https://www.tradingview.com/v/amRCTgFw/
# call example 
# NadarayaWatsonEstimator2(dataframe, source = 'close', bandwidth = 8, candlesCount = 500)
# buy (dataframe['NWE'] == 1) & 
# sell (dataframe['NWE'] == -1) & 
def NadarayaWatsonEstimator2(dtloc, source = 'close', bandwidth = 8, candlesCount = 500):
    y1 = 0.0
    y2 = 0.0
    y1_d = 0.0
    dtloc['NWE'] = 0
    for i in range(candlesCount):
        sum = 0.0
        sumw = 0.0
        for j in range(candlesCount):
            w = math.exp(-(math.pow(i-j,2)/(bandwidth*bandwidth*2)))
            sum = sum + (dtloc[source].iloc[-(j+1)] * w)
            sumw = sumw + w
        y2 = sum/sumw
        d = y2 - y1
        nweresult = 0
        if (d > 0 and y1_d < 0):
            nweresult = 1
        if (d < 0 and y1_d > 0):
            nweresult = -1
        dtloc['NWE'].iloc[-(i+1)] = nweresult
        y1 = y2
        y1_d = d

# #"Optimized Trend Tracker
# def getOTT(dtloc, source = 'close',  OTTindicatorType = 'EMA', length = 14, OTTPeriod = 2, OTTPercent = 1.4):
    
#     dtOTT = dtloc.copy()
#     dtOTT['MAvg'] = get(dtloc, indicatorType = OTTindicatorType, source = source, length = length)
#     dtOTT['fark'] = dtOTT['MAvg'] * OTTPercent *0.01
#     dtOTT['longStop'] = dtOTT['MAvg'] - dtOTT['fark']
#     dtOTT['longStopPrev'] = dtOTT['longStop'].shift(1).fillna(dtOTT['longStop'])
#     dtOTT['longStop'] = np.where(dtOTT['MAvg'] > dtOTT['longStopPrev'],  np.where(dtOTT['longStop'] > dtOTT['longStopPrev'], dtOTT['longStop'] , dtOTT['longStopPrev']) ,  dtOTT['longStop'])

#     dtOTT['shortStop'] = dtOTT['MAvg'] + dtOTT['fark']
#     dtOTT['shortStopPrev'] = dtOTT['shortStop'].shift(1).fillna(dtOTT['shortStop'])
#     dtOTT['shortStop'] = np.where(dtOTT['MAvg'] < dtOTT['shortStopPrev'],  np.where(dtOTT['shortStop'] > dtOTT['shortStopPrev'], dtOTT['shortStopPrev'] , dtOTT['shortStop']) ,  dtOTT['shortStop'])

#     dtOTT['dir'] = np.where(dtOTT['MAvg'] > dtOTT['shortStopPrev'],  1,  np.where(dtOTT['MAvg'] < dtOTT['longStopPrev'], -1,  1 )

#     dtOTT['MT'] = np.where(dtOTT['dir'] == 1,  dtOTT['longStop'], dtOTT['shortStop'])

#     dtOTT['OTT'] = np.where(dtOTT['MAvg'] > dtOTT['MT'],  dtOTT['MT'] *(200+OTTPercent)/200   ,  dtOTT['MT'] *(200-OTTPercent)/200 )
    
#     return dtOTT['OTT']


# def getSSLExit(dtloc, sslExitType = 'HMA', sslExitlength = 15):
#     dfExit = dtloc.copy()
    
#     dfExit['exitHigh']  = get(dfExit, indicatorType = sslExitType, source = 'high', length = sslExitlength)
#     dfExit['exitLow']   = get(dfExit,  indicatorType = sslExitType, source = 'low' , length = sslExitlength)
        
#     calc_hlv3_hlv3 = 0
#     def calc_hlv3(dfr):
#         global calc_hlv3_hlv3
#         try:
#             calc_hlv3_hlv3
#         except NameError:
#             calc_hlv3_hlv3 = -1
#         if math.isnan(calc_hlv3_hlv3) == True:
#             calc_hlv3_hlv3 = -1  
#         if dfr['close'] >  dfr['exitHigh']:
#             calc_hlv3_hlv3 = 1
#         elif dfr['close'] <  dfr['exitLow']:
#             calc_hlv3_hlv3 = -1
#         return calc_hlv3_hlv3
#     dfExit['hlv3'] = dfExit.apply(calc_hlv3, axis = 1)
#     dfExit['sslExit'] = np.where(dfExit['hlv3'] < 0,dfExit['exitHigh'],dfExit['exitLow'])

#     return dfExit['sslExit'], dfExit['exitHigh'], dfExit['exitLow']



#!/bin/bash
## declare an array variable

declare -a array=("SOL/USDT" "WIN/USDT" "LINK/USDT" "TRX/USDT" "CHZ/USDT" "AAVE/USDT" "THETA/USDT" "AVAX/USDT" "FIL/USDT" "MKR/USDT" "UTK/USDT" "TKO/USDT" "BTT/USDT" "ONT/USDT"  "ETH/USDT" "WAVES/USDT" "CAKE/USDT" "LUNA/USDT" "NEO/USDT" "DENT/USDT" "XRP/USDT" "ETC/USDT" "BCH/USDT" "LTC/USDT" "ADA/USDT" "TWT/USDT" "DOT/USDT" "TLM/USDT" "BTC/USDT" "IRIS/USDT" "HOT/USDT" "FTM/USDT" "XLM/USDT" "VET/USDT" "DOGE/USDT")

# get length of an array
arraylength=${#array[@]}


varStratName="MultiMA_TSL"
varEpohs="750"

# use for loop to read all values and indexes
for (( i=0; i<${arraylength}; i++ ));
do
  freqtrade hyperopt  --hyperopt-loss SortinoHyperOptLoss  \
   		      --spaces buy sell \
		      --strategy  ${varStratName} \
		      --config configTEST.json \
                      --timerange=20210912- \
                      -e ${varEpohs} \
                      --timeframe 5m \
                      -j 20 \
                      -p  ${array[$i]} #\

  var_input="user_data/strategies/${varStratName}.json"
  var_file=${array[$i]%/USDT}
  var_output="user_data/strategies/${varStratName}_${var_file}.json"
  mv $var_input $var_output
done

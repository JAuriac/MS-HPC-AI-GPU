#!/bin/bash

csvfile=csv/GPU.csv
#init=flowAroundCylinder.ini

echo "Running"

mkdir -p log
rm $csvfile

echo "nxny;maxIter;elapsed;total_cost" > $csvfile

array=( 256 512 1024 2048 4096 8192 16384 ) #nx
array2=( 64 128 256 512 1024 2048 4096 ) #ny

for i in "${!array[@]}"; do
  nx=${array[i]}
  ny=${array2[i]}

  echo "Loop $i... nx=$nx ny=$ny"

  sed -i "12s/.*/nx=$nx/" flowAroundCylinder.ini
  sed -i "13s/.*/ny=$ny/" flowAroundCylinder.ini

  ./lbmFlowAroundCylinder ./flowAroundCylinder.ini > tmp_log

  line=$(grep Results tmp_log)

  nxny=$(echo $line | cut -d ';' -f2)
  maxIter=$(echo $line | cut -d ';' -f3)
  elapsed=$(echo $line | cut -d ';' -f4)
  total_cost=$(echo $line | cut -d ';' -f5)

  echo "${nxny};${maxIter};${elapsed};${total_cost}" >> $csvfile
done

rm tmp_log

echo "Done"

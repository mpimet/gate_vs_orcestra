#!/bin/bash

thisdir=$(pwd)

cd GATE_Radiosonde_Data/3.31.02.101-3.33.02.101_19740601-19740930_v3

for infile in `ls -1 | grep -v .sh`; do
  echo $infile
  ${thisdir}/GATE_other_radiosonde.x $infile
done
    

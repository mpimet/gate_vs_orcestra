#!/bin/bash

thisdir=$(pwd)

# cd ...../GATE_AND_COMM_SHIPS/$1
cd ...../METEOR/$1

for infile in `ls -1 | grep -v .sh`; do
  ${thisdir}/GATEdship_$1.x $infile
done
    

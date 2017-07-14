#!/bin/bash

if [ $# == 0 ]
  then
    echo "Usage: ./run_benchmark.sh exe output_dir"
    exit
fi

# EXEC=./main2d.pgi.TPROF.CUDA.ex
# EXEC=./main2d.gnu.TPROF.ex
EXEC=$1
INPUT=inputs_2d
OUTPUT=run.log

mkdir -p $2 && cp {$INPUT,$EXEC,advance_2d.F90} $2 && cd $2

# have changed directory to the benchmark directory 
$EXEC $INPUT &>$OUTPUT



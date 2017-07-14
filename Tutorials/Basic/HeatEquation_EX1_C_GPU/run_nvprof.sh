#!/bin/bash

if [ $# == 0 ]
  then
    echo "Usage: ./run_nvprof.sh application_name output_dir"
    exit
fi

APPLICATION=$1
OUTPUTDIR=$2
INPUT=inputs_2d
OUTPUT=main.nvprof

mkdir -p $OUTPUTDIR && cp {$INPUT,$APPLICATION,advance_2d.F90} $OUTPUTDIR && cd $OUTPUTDIR
wait

# instruct the profiling tool to disable profiling at the start of the application
OPTIONS=\ --profile-from-start\ on\ --export-profile\ $OUTPUT 
# OPTIONS=\ --analysis-metrics

nvprof $OPTIONS $APPLICATION $INPUT &>run.log
# echo $OPTIONS 
# echo $APPLICATION 
# echo $INPUT



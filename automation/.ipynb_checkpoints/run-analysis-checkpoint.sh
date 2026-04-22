#!/bin/bash

# run this script from the root (folder): ./automation/run-analysis.sh pathPARAMSFILE/PARAMSFILE.sh

DIR="$(dirname "${BASH_SOURCE[0]}")"
study_path="$(basename $DIR)"

if [ $# -eq 0 ]
  then
     params_file=$DIR/noveltysearch.sh
  else
    params_file=$1
fi

source $params_file

comparison='basic_plots'
#python experiments/${study_path}/snapshots_bests.py $study $experiments $tfs $runs $generations $out_path;
#python experiments/${study_path}/bests_snap_draw.py $study $experiments $runs $generations $out_path;

# pip install papermill
papermill "experiments/${study_path}/analysis.ipynb" "${out_path:-experiments/${study_path}}/analysis-executed.ipynb" \
  -p study "$study" -p experiments "$experiments" -p runs "$runs" -p generations "$generations"


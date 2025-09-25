#!/bin/bash

# 定义被试列表
subjects=("sub-01" "sub-02" "sub-03" "sub-05" "sub-06" "sub-07" "sub-08" "sub-09")

# 循环处理每个被试
for sub in "${subjects[@]}"; do
    (
    echo "Processing $sub..."
    python '/Users/derek/Desktop/Proj-w-YCTY/kymata-core/kymata/invokers/run_gridsearch.py' \
    --config "/Users/derek/Desktop/Proj-w-YCTY/kymata-core/dataset_config/dataset_ecog.yaml" \
    --input-stream 'auditory' \
    --emeg-dir 'ecogprep' \
    --transform-name STL \
    --transform-path "stimulisig/stimulisig_podcast" \
    --seconds-per-split 1 \
    --n-splits 1798 \
    --save-name "gridsearch_trial_${sub}" \
    --single-participant-override "$sub"
    
    echo "$sub completed!"
    ) &
done

wait
echo "All participants processed!"
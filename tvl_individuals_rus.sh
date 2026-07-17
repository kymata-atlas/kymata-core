#!/bin/bash

declare -a participants=("meg15_0045" 
                "meg15_0051" 
                "meg15_0054"
                "meg15_0055"
                "meg15_0056"
                "meg15_0058"
                "meg15_0060"
                "meg15_0065"
                "meg15_0066"
                "meg15_0070"
                "meg15_0071"
                "meg15_0072"
                "meg15_0079"
                "meg15_0081"
                "meg15_0082")

for participant in "${participants[@]}"; do
  poetry run python -m kymata.invokers.run_gridsearch \
    --config dataset3.yaml --input-stream auditory --transform-path 'predicted_function_contours/audio/GMSloudness/stimulisig' \
    --transform-name IL STL IL1 IL2 IL3 IL4 IL5 IL6 IL7 IL8 IL9 --plot-top-channels --emeg-dir 'interim_preprocessing_files/3_trialwise_sensorspace/evoked_data/' \
    --save-expression-set-location '/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/tvl/individuals' \
    --save-plot-location '/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/tvl/individuals' \
    --single-participant-override "${participant}" \
    --save-name "rus_${participant}"
done

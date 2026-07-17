#!/bin/bash

declare -a participants=('pilot_01'
                'pilot_02'
                'pilot_03'
                'pilot_04'
                'participant_01'
                'participant_02'
                'participant_03'
                'participant_05'
                'participant_06'
                'participant_07'
                "participant_08"
                "participant_09"
                "participant_10"
                "participant_11"
                "participant_12")

for participant in "${participants[@]}"; do
  poetry run python -m kymata.invokers.run_gridsearch \
    --config dataset4.1.yaml --input-stream auditory --transform-path 'predicted_function_contours/GMSloudness/stimulisig' \
    --transform-name IL STL IL1 IL2 IL3 IL4 IL5 IL6 IL7 IL8 IL9 --plot-top-channels --emeg-dir 'interim_preprocessing_files/3_trialwise_sensorspace/evoked_data/eng_rus/' \
    --save-expression-set-location '/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/tvl/individuals' \
    --save-plot-location '/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/tvl/individuals' \
    --single-participant-override "${participant}" \
    --save-name "en_rus_${participant}"
done

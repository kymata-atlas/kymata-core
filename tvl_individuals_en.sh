#!/bin/bash

declare -a participants=('pilot_01'
                'pilot_02'
                'participant_01'
                'participant_01b'
                'participant_02'
                'participant_03'
                'participant_04'
                'participant_05'
                'participant_07'
                'participant_08'
                'participant_09'
                'participant_10'
                'participant_11'
                'participant_12'
                'participant_13'
                'participant_14'
                'participant_15'
                'participant_16'
                'participant_17'
                'participant_18'
                'participant_19')

for participant in "${participants[@]}"; do
  poetry run python -m kymata.invokers.run_gridsearch \
    --config dataset4.yaml --input-stream auditory --transform-path 'predicted_function_contours/GMSloudness/stimulisig' \
    --transform-name IL STL IL1 IL2 IL3 IL4 IL5 IL6 IL7 IL8 IL9 --plot-top-channels --emeg-dir 'interim_preprocessing_files/3_trialwise_sensorspace/evoked_data/' \
    --save-expression-set-location '/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/tvl/individuals' \
    --save-plot-location '/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/tvl/individuals' \
    --single-participant-override "${participant}" \
    --save-name "en_${participant}"
done

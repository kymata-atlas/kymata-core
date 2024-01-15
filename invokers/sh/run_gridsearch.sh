python invokers/run_gridsearch.py \
  --base-dir "/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/" \
  --data-path "intrim_preprocessing_files/3_trialwise_sensorspace/evoked_data" \
  --function-path "predicted_function_contours/GMSloudness/stimulisig" \
  --function-name "d_IL2" \
  --emeg-file "participant_01-ave" \
  --inverse-operator "intrim_preprocessing_files/4_hexel_current_reconstruction/inverse-operators" \
  --overwrite

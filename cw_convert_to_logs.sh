in="/Users/cai/Dox/Work/Kymata lab/Code/kymata-core/kymata-core-data/tianyi/sensor/expression_sets_split/"
out="/Users/cai/Dox/Work/Kymata lab/Code/kymata-core/kymata-core-data/tianyi/sensor/logs_split/"

declare -a modalities=("eeg" "meg")

for i in {0..32}; do
  for emeg in "${modalities[@]}"; do
    poetry run python -m kymata.invokers.invoker_exp2log \
      -o "${out}" -i "${in}/layer${i}_4095_gridsearch_${emeg}.nkg"
  done
done

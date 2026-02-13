in="/Users/cai/Dox/Work/Kymata lab/Code/kymata-core/kymata-core-data/tianyi/sensor/salmonn_7b_word/expression_set"
out="/Users/cai/Dox/Work/Kymata lab/Code/kymata-core/kymata-core-data/tianyi/sensor/expression_sets_split/"

for i in {0..32}; do
  poetry run python -m kymata.invokers.split_meg_eeg -o "${out}" -i "${in}/layer${i}/layer${i}_4095_gridsearch.nkg"
done

from pathlib import Path
from os import path
from kymata.io.nkg import load_expression_set
from kymata.plot.plot import expression_plot

# template invoker for printing out expression set .nkgs

path_to_nkg_files = Path(Path(path.abspath("")).parent, "kymata-toolbox/kymata-toolbox-data", "output/whisper/decoder_k")

# expression_data = load_expression_set(Path( path_to_nkg_files, "model.encoder.conv1_511_gridsearch.nkg"))
# expression_data = load_expression_set(Path( path_to_nkg_files, "model.encoder.conv2_511_gridsearch.nkg"))
# expression_data = load_expression_set(Path( path_to_nkg_files, "model.encoder.layers.0.final_layer_norm_511_gridsearch.nkg"))
# expression_data = load_expression_set(Path( path_to_nkg_files, "model.encoder.layers.1.final_layer_norm_511_gridsearch.nkg"))
# expression_data = load_expression_set(Path( path_to_nkg_files, "model.encoder.layers.2.final_layer_norm_511_gridsearch.nkg"))
# expression_data = load_expression_set(Path( path_to_nkg_files, "model.encoder.layers.3.final_layer_norm_511_gridsearch.nkg"))
# expression_data = load_expression_set(Path( path_to_nkg_files, "model.encoder.layers.4.final_layer_norm_511_gridsearch.nkg"))
# expression_data = load_expression_set(Path( path_to_nkg_files, "model.encoder.layers.5.final_layer_norm_511_gridsearch.nkg"))

# expression_data = load_expression_set(Path( path_to_nkg_files, "model.decoder.layers.0.encoder_attn.v_proj_511_gridsearch.nkg"))
# expression_data = load_expression_set(Path( path_to_nkg_files, "model.decoder.layers.1.encoder_attn.v_proj_511_gridsearch.nkg"))
# expression_data = load_expression_set(Path( path_to_nkg_files, "model.decoder.layers.2.encoder_attn.v_proj_511_gridsearch.nkg"))
# expression_data = load_expression_set(Path( path_to_nkg_files, "model.decoder.layers.3.encoder_attn.v_proj_511_gridsearch.nkg"))
# expression_data = load_expression_set(Path( path_to_nkg_files, "model.decoder.layers.4.encoder_attn.v_proj_511_gridsearch.nkg"))
# expression_data = load_expression_set(Path( path_to_nkg_files, "model.decoder.layers.5.encoder_attn.v_proj_511_gridsearch.nkg"))

# expression_data = load_expression_set(Path( path_to_nkg_files, "model.decoder.layers.0.encoder_attn.k_proj_511_gridsearch.nkg"))
# expression_data = load_expression_set(Path( path_to_nkg_files, "model.decoder.layers.1.encoder_attn.k_proj_511_gridsearch.nkg"))
# expression_data = load_expression_set(Path( path_to_nkg_files, "model.decoder.layers.2.encoder_attn.k_proj_511_gridsearch.nkg"))
# expression_data = load_expression_set(Path( path_to_nkg_files, "model.decoder.layers.3.encoder_attn.k_proj_511_gridsearch.nkg"))
# expression_data = load_expression_set(Path( path_to_nkg_files, "model.decoder.layers.4.encoder_attn.k_proj_511_gridsearch.nkg"))
expression_data = load_expression_set(Path( path_to_nkg_files, "model.decoder.layers.5.encoder_attn.k_proj_511_gridsearch.nkg"))


expression_plot(expression_data, ylim=-400, xlims=(-200, 800), save_to=Path(Path(path.abspath("")).parent, "kymata-toolbox/kymata-toolbox-data", "output/expression_plot/decoderk5.jpg"), show_legend=False)
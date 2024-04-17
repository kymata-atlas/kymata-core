from pathlib import Path
from os import path
from kymata.io.nkg import load_expression_set
from kymata.plot.plot import expression_plot

# template invoker for printing out expression set .nkgs

path_to_nkg_files = Path(Path(path.abspath("")).parent, "kymata-toolbox/kymata-toolbox-data", "output/whisper/encoder_all")

expression_data = load_expression_set(Path( path_to_nkg_files, "model.encoder.conv1_511_gridsearch.nkg"))
# expression_data += load_expression_set(Path( path_to_nkg_files, "neurogram_mr_gridsearch.nkg"))

expression_plot(expression_data, ylim=-400, xlims=(-200, 800), save_to=Path(Path(path.abspath("")).parent, "kymata-toolbox/kymata-toolbox-data", "output/expression_plot/conv1.jpg"), show_legend=False)
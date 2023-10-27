from pathlib import Path

from kymata.plot.plotting import expression_plot
from kymata.io.matlab import load_matab_expression_files

sample_data_dir = Path(Path(__file__).parent.parent, "data", "sample-data")
expression_data = load_matab_expression_files(
    function_name="hornschunck_horizontalPosition",
    lh_file=Path(sample_data_dir,
                 "hornschunck_horizontalPosition_lh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
    rh_file=Path(sample_data_dir,
                 "hornschunck_horizontalPosition_rh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
    flipped_lh_file=Path(sample_data_dir,
                         "hornschunck_horizontalPosition-flipped_lh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
    flipped_rh_file=Path(sample_data_dir,
                         "hornschunck_horizontalPosition-flipped_rh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
)
expression_data += load_matab_expression_files(
    function_name="hornschunck_horizontalVelocity",
    lh_file=Path(sample_data_dir,
                 "hornschunck_horizontalVelocity_lh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
    rh_file=Path(sample_data_dir,
                 "hornschunck_horizontalVelocity_rh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
    flipped_lh_file=Path(sample_data_dir,
                         "hornschunck_horizontalVelocity-flipped_lh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
    flipped_rh_file=Path(sample_data_dir,
                         "hornschunck_horizontalVelocity-flipped_rh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
)
expression_data += load_matab_expression_files(
    function_name="ins_loudness",
    lh_file=Path(sample_data_dir, "ins_loudness_lh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
    rh_file=Path(sample_data_dir, "ins_loudness_rh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
    flipped_lh_file=Path(sample_data_dir,
                         "ins_loudness-flipped_lh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
    flipped_rh_file=Path(sample_data_dir,
                         "ins_loudness-flipped_rh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
)
expression_plot(expression_data, include_functions=["hornschunck_horizontalVelocity"])

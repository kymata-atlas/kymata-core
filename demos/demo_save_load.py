from pathlib import Path

from kymata.entities.expression import ExpressionSet
from kymata.io.matlab import load_matab_expression_files

# set location of tutorial data
sample_data_dir = Path(Path(__file__).parent.parent, "data", "sample-data")

# Load in an existing expression set object
expression_data_kymata_mirror = ExpressionSet.load(from_path=Path(sample_data_dir, "kymata_mirror_Q3_2023_expression_endtable.nkg"))

# Create new expression set object for the new results (or you can just add to an existing expressionSet
# directly using '+=' ).
expression_data_new_results = load_matab_expression_files(
    function_name="ins_loudness_2020",
    lh_file=Path(sample_data_dir, "GMloudness_lh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
    rh_file=Path(sample_data_dir, "GMloudness_rh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
)
expression_data_new_results += load_matab_expression_files(
    function_name="delta_ins_loudness_tonotop_chan1_2020",
    lh_file=Path(sample_data_dir, "GMloudness_tonotop_82dB__d_ins_loudness_tonop_chan1__lh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
    rh_file=Path(sample_data_dir, "GMloudness_tonotop_82dB__d_ins_loudness_tonop_chan1__rh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
)

# You can add two ExpressionSets together
expression_data_extended = expression_data_kymata_mirror + expression_data_new_results

# Save new expressionSet for use again in the future.
expression_data_extended.save(to_path=Path(sample_data_dir, "kymata_mirror_Q3_2023_expression_endtable_extended.nkg"), overwrite=False)


from pathlib import Path

from kymata.entities.expression import load_matab_expression_files, ExpressionSet

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

expression_data.save("/Users/cai/Desktop/temp.nkg", overwrite=True)
expression_data2 = ExpressionSet.load(from_path=Path("/Users/cai/Desktop/temp.nkg"))

print(expression_data == expression_data2)

expression_data2.save(to_path=Path("/Users/cai/Desktop/temp2.nkg"), overwrite=True)

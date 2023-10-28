from pathlib import Path

from kymata.plot.plotting import expression_plot
from kymata.entities.expression import ExpressionSet


# set location of tutorial data
sample_data_dir = Path(Path(__file__).parent.parent, "data", "sample-data")

# create new expression set object and add to it
expression_data_kymata_mirror = ExpressionSet.load(from_path=Path(sample_data_dir, "kymata_mirror_Q3_2023_expression_endtable.nkg"))

# print the names of all available functions in the expressionSet object
print(expression_data_kymata_mirror.functions)

# plot everything, with everything model selected against each other
expression_plot(expression_data_kymata_mirror)

# only compare a subset of functions (e.g. colour functions), and print them all.
# Note that 'CIELAB a*' and 'CIELAB L' are not significant, and so will not turn up.
expression_plot(expression_data_kymata_mirror[
                    'CIECAM02 A',
                    'CIECAM02 a',
                    'CIELAB a*',
                    'CIELAB L'
                ])

# Only compare a subset of functions, and print just one of them
expression_plot(expression_data_kymata_mirror[
                    'CIECAM02 A',
                    'CIECAM02 a',
                    'CIELAB a*',
                    'CIELAB L'
                ], show_only=["CIECAM02 A"])

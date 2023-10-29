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
                ], show_only=[
                    "CIECAM02 A"
                ])

# Override colours (using hexcodes) so that all touch are cyan, all hearing orange, and all visual purple
expression_plot(expression_data_kymata_mirror[
                    'vibration detection (RH-Th/P/M)',
                    'vibration detection (LH-Th/P/M)',
                    'CIECAM02 A',
                    'CIECAM02 a',
                    'Heeger horizontal velocity',
                    'Heeger horizontal ME GP1',
                    'Heeger horizontal ME GP2',
                    'Heeger horizontal ME GP3',
                    'Heeger horizontal ME GP4',
                    'TVL loudness (short-term)', 
                    'TVL loudness (instantaneous)', 
                    'TVL loudness chan 1 (instantaneous)', 
                    'TVL loudness chan 2 (instantaneous)', 
                    'TVL loudness chan 3 (instantaneous)', 
                    'TVL loudness chan 4 (instantaneous)', 
                    'TVL loudness chan 5 (instantaneous)', 
                    'TVL loudness chan 6 (instantaneous)', 
                    'TVL loudness chan 7 (instantaneous)',
                    'TVL loudness chan 8 (instantaneous)',
                    'TVL loudness chan 9 (instantaneous)',
                ], color = {
                    'vibration detection (RH-Th/P/M)': '#21d4ca',
                    'vibration detection (LH-Th/P/M)': '#21d4ca',
                    'CIECAM02 A': '#af90e3',
                    'CIECAM02 a': '#af90e3',
                    'Heeger horizontal velocity': '#af90e3',
                    'Heeger horizontal ME GP1': '#af90e3',
                    'Heeger horizontal ME GP2': '#af90e3',
                    'Heeger horizontal ME GP3': '#af90e3',
                    'Heeger horizontal ME GP4': '#af90e3',
                    'TVL loudness (short-term)': '#f1b37e',
                    'TVL loudness (instantaneous)': '#f1b37e',
                    'TVL loudness chan 1 (instantaneous)': '#f1b37e',
                    'TVL loudness chan 2 (instantaneous)': '#f1b37e',
                    'TVL loudness chan 3 (instantaneous)': '#f1b37e',
                    'TVL loudness chan 4 (instantaneous)': '#f1b37e',
                    'TVL loudness chan 5 (instantaneous)': '#f1b37e',
                    'TVL loudness chan 6 (instantaneous)': '#f1b37e',
                    'TVL loudness chan 7 (instantaneous)': '#f1b37e',
                    'TVL loudness chan 8 (instantaneous)': '#f1b37e',
                    'TVL loudness chan 9 (instantaneous)': '#f1b37e'
                  })
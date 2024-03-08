from pathlib import Path

from kymata.datasets.sample import KymataMirror2023Q3Dataset
from kymata.io.config import load_config
from kymata.plot.color import gradient_color_dict, constant_color_dict
from kymata.plot.plot import expression_plot


# template invoker for printing out expression set .nkgs
def main():
    config = load_config(str(Path(Path(__file__).parent.parent, "kymata", "config", "dataset3.yaml")))

    expression_data = KymataMirror2023Q3Dataset().to_expressionset()

    expression_plot(expression_data,
                    show_only=[
                        'TVL loudness chan 1 (instantaneous)',
                        'TVL loudness chan 2 (instantaneous)',
                        'TVL loudness chan 3 (instantaneous)',
                        'TVL loudness chan 4 (instantaneous)',
                        'TVL loudness chan 5 (instantaneous)',
                        'TVL loudness chan 6 (instantaneous)',
                        'TVL loudness chan 7 (instantaneous)',
                        'TVL loudness chan 8 (instantaneous)',
                        'TVL loudness chan 9 (instantaneous)',
                        'TVL loudness (instantaneous)',
                        'TVL loudness (short-term)',

                    ],
                    color=gradient_color_dict([
                        'TVL loudness chan 1 (instantaneous)',
                        'TVL loudness chan 2 (instantaneous)',
                        'TVL loudness chan 3 (instantaneous)',
                        'TVL loudness chan 4 (instantaneous)',
                        'TVL loudness chan 5 (instantaneous)',
                        'TVL loudness chan 6 (instantaneous)',
                        'TVL loudness chan 7 (instantaneous)',
                        'TVL loudness chan 8 (instantaneous)',
                        'TVL loudness chan 9 (instantaneous)',
                        ], start_color="blue", stop_color="purple")
                    | constant_color_dict([
                        'TVL loudness (instantaneous)',
                    ], color="red")
                    | constant_color_dict([
                        'TVL loudness (short-term)1',
                    ], color="pink"),
                    minimap_config=config,
                    minimap_view="lateral",
                    minimap_surface="inflated",
                    )

if __name__ == '__main__':
    main()

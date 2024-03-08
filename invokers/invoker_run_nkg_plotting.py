from pathlib import Path

from kymata.datasets.sample import KymataMirror2023Q3Dataset
from kymata.io.config import load_config
from kymata.plot.plot import expression_plot


# template invoker for printing out expression set .nkgs
def main():
    config = load_config(str(Path(Path(__file__).parent.parent, "kymata", "config", "dataset4.yaml")))

    expression_data = KymataMirror2023Q3Dataset().to_expressionset()

    expression_plot(expression_data,
                    show_only=[
                        'vibration detection (LH-Th/P/M)',
                    ],
                    minimap_config=config,
                    minimap_view="lateral",
                    minimap_surface="inflated",
                    )

if __name__ == '__main__':
    main()

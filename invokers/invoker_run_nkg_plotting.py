from pathlib import Path
from os import path

from kymata.io.nkg import load_expression_set
from kymata.plot.plot import expression_plot

# template invoker for printing out expression set .nkgs
def main():
    path_to_nkg_files = Path(Path(path.abspath("")).parent, "kymata-toolbox-data", "output")

    expression_data  = load_expression_set(Path( path_to_nkg_files, "combined_TVL_gridsearch.nkg"))

    fig = expression_plot(expression_data,
                          minimap=True,
                          minimap_view="lateral",
                          minimap_surface="inflated",
                          )

    fig.show()


if __name__ == '__main__':
    main()

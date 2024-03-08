from pathlib import Path

from kymata.io.config import load_config, get_root_dir
from kymata.preproc.hexel_current_estimation import create_forward_model_and_inverse_solution, create_hexel_morph_maps


def main():
    config = load_config(str(Path(Path(__file__).parent.parent, "kymata", "config", "dataset4.yaml")))

    data_root_dir = get_root_dir(config)

    # create_current_estimation_prerequisites(data_root_dir, config=config)

    create_forward_model_and_inverse_solution(data_root_dir, config=config)

    create_hexel_morph_maps(data_root_dir, config=config)


if __name__ == '__main__':
    main()

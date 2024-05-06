from pathlib import Path

from kymata.io.config import load_config, get_root_dir
from kymata.preproc.hexel_current_estimation import create_current_estimation_prerequisites, create_forward_model_and_inverse_solution, create_hexel_morph_maps


def main(config_filename: str):
    config = load_config(str(Path(Path(__file__).parent.parent, "dataset_config", config_filename)))

    data_root_dir = get_root_dir(config)

    create_current_estimation_prerequisites(data_root_dir, config=config)

    create_forward_model_and_inverse_solution(data_root_dir, config=config)

    create_hexel_morph_maps(data_root_dir, config=config)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the appropriate dataset config .yaml file", default="dataset4.yaml")
    args = parser.parse_args()

    main(config_filename=args.config)

import argparse
from pathlib import Path
from logging import getLogger, basicConfig, INFO

from kymata.datasets.data_root import data_root_path
from kymata.io.cli import log_message, date_format
from kymata.preproc.source import load_emeg_pack

_default_output_dir = Path(data_root_path(), "output")

_logger = getLogger(__name__)


def main():

    _default_output_dir.mkdir(exist_ok=True, parents=False)

    parser = argparse.ArgumentParser(description='Gridsearch Params')
    parser.add_argument('--base-dir', type=Path, default=Path('/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/'),
                        help='base data directory')
    parser.add_argument('--emeg-dir', type=str, default='intrim_preprocessing_files/3_trialwise_sensorspace/evoked_data/',
                        help='emeg directory, relative to base dir')
    parser.add_argument('--morph', action="store_true",
                        help="Morph hexel data to fs-average space prior to running gridsearch. "
                             "Only has an effect if an inverse operator is specified.")
    parser.add_argument('--ave-mode', type=str, default="ave", choices=["ave", "add"],
                        help='`ave`: average over the list of repetitions. `add`: treat them as extra data.')
    parser.add_argument('--inverse-operator-dir', type=str, default=None,
                        help='inverse solution path')
    parser.add_argument('--inverse-operator-suffix', type=str, default="_ico5-3L-loose02-cps-nodepth-fusion-inv.fif",
                        help='inverse solution suffix')
    args = parser.parse_args()

    participants = [
        'pilot_01',
        'pilot_02',
        # 'participant_01',
        # 'participant_01b',
        # 'participant_02',
        # 'participant_03',
        # 'participant_04',
        # 'participant_05',
        # 'participant_07',
        # 'participant_08',
        # 'participant_09',
        # 'participant_11',
        # 'participant_12',
        # 'participant_13',
    ]

    # TODO: move ave-vs-reps choice up to the function interface
    reps = [f'_rep{i}' for i in range(8)] + ['-ave']
    emeg_filenames = [
        p + r
        for p in participants
        for r in reps[-1:]
    ]

    if (len(emeg_filenames) > 1) and (not args.morph) and (args.ave_mode == "ave") and (args.inverse_operator_dir is not None):
        raise ValueError(
            f"Averaging source-space results without morphing to a common space. "
            f"If you are averaging over multiple participants you must morph to a common space.")

    # Load data
    emeg_path = Path(args.base_dir, args.emeg_dir)
    morph_dir = Path(args.base_dir, "intrim_preprocessing_files", "4_hexel_current_reconstruction", "morph_maps")
    _logger.info(f"Loading EMEG pack from {emeg_filenames}")
    emeg_values, ch_names = load_emeg_pack(emeg_filenames,
                                           emeg_dir=emeg_path,
                                           morph_dir=morph_dir,
                                           use_morph=args.morph,
                                           need_names=True,
                                           ave_mode=args.ave_mode,
                                           inverse_operator_dir=args.inverse_operator_dir,
                                           inverse_operator_suffix= args.inverse_operator_suffix,
                                           p_tshift=None,
                                           )


if __name__ == '__main__':
    basicConfig(format=log_message, datefmt=date_format, level=INFO)
    _logger.info("Starting..")
    main()
    _logger.info("Done!")

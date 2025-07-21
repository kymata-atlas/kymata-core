from logging import getLogger
from os import path
from pathlib import Path
import numpy as np
import nibabel as nib

import mne
import mne.bem
import scipy
from scipy.sparse import csr_matrix

_logger = getLogger(__file__)


def create_current_estimation_prerequisites(data_root_dir, config: dict):
    """
    Copy the structurals to the local Kymata folder,
    create the surfaces, the boundary element model solutions, and the source space
    """

    list_of_participants = config["participants"]
    dataset_directory_name = config["dataset_directory_name"]
    interim_preprocessing_directory_name = Path(
        data_root_dir, dataset_directory_name, "interim_preprocessing_files"
    )
    mri_structural_type = config['mri_structural_type']
    mri_structurals_directory = Path(
        data_root_dir, dataset_directory_name, config["mri_structurals_directory"]
    )

    """    

    # <--------------------Command Line-------------------------->

    # Set location in the Kymata Project directory
    # where the converted MRI structurals will reside, and create the folder structure
    $ freesurfer_6.0.0 (may need to setup freesurfer first as follows if FREESURFER_HOME not set persistently)
    $ export FREESURFER_HOME=/imaging/local/software/freesurfer/6.0.0/x86_64
    $ source $FREESURFER_HOME/SetUpFreeSurfer.sh

    $ setenv SUBJECTS_DIR /imaging/projects/cbu/kymata/data/dataset_4-english_narratives/raw_mri_structurals/
    $ the previous command shoule this in bash:
    $ export SUBJECTS_DIR=/imaging/projects/cbu/kymata/data/dataset_5-tactile_fingertips/raw_mri_structurals
    for all participants:
        $ mksubjdirs participant_01 # note - this appears to ignore SUBJECTS_DIR and uses the folder you are in.

    # Load the fsaverage mesh
    $ cp -r $FREESURFER_HOME/subjects/fsaverage $SUBJECTS_DIR/fsaverage
    
    # create a one-off source space of fsaverage to morph to later.
    src = mne.setup_source_space(
        "fsaverage", spacing="ico5", subjects_dir=mri_structurals_directory, verbose=True
    )
    mne.write_source_spaces(Path(interim_preprocessing_directory_name,
                                     "4_hexel_current_reconstruction",
                                     "src_files",
                                     "fsaverage_ico5-src.fif"), src)
    
    # todo - we were using the "aparc.DKTatlas40" atlas - is this the same as the Desikan-Killiany Atlas, and aparc.DKTatlas?
    # (?h.aparc.annot)? https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation. And if so, do we
    # need to add it to the fsaverage folder for fsaverage? (I don't think it is used here, although we do
    # use it in Kymata web. If so, remove this section). i.e.
    #$ cp $FREESURFER_HOME/average/rh.DKTatlas40.gcs  $SUBJECTS_DIR/fsaverage/rh.DKTatlas40.gcs
    #$ cp $FREESURFER_HOME/average/lh.DKTatlas40.gcs  $SUBJECTS_DIR/fsaverage/lh.DKTatlas40.gcs
    #$ mris_ca_label -orig white -novar fsaverage rh sphere.reg $SUBJECTS_DIR/fsaverage/rh.DKTatlas40.gcs $SUBJECTS_DIR/fsaverage/label/rh.aparc.DKTatlas40.annot
    #$ mris_ca_label -orig white -novar fsaverage lh sphere.reg $SUBJECTS_DIR/fsaverage/lh.DKTatlas40.gcs $SUBJECTS_DIR/fsaverage/label/lh.aparc.DKTatlas40.annot

    # move data across from the MRIdata folder to the local
    # directory, so freesurfer can find it - also convert from dcm to .mgz
    for participant in participants
        $ mri_convert /mridata/cbu/CBU230790_MEG23008/20231102_130449/Series005_CBU_MPRAGE_32chn/1.3.12.2.1107.5.2.43.67035.202311021312263809335255.dcm $SUBJECTS_DIR/participant_01/mri/orig/001.mgz

    # creates suitable T1, meshes and labels
    for participant in participants
        $ recon-all -s participant_01 -all

    # creates suitable T1, meshes and labels... but using python?
    for participant in participants
        The source space (downsampled version of the cortical surface in Freesurfer), which will be saved in a file ending in *-src.fif, which can be read into Matlab using mne_read_source_spaces.


        mne.viz.plot_alignment()
        mne.viz.plot_bem(),

        # create labels for these individuals, for Kymata we prefer the aparc.DKTatlas40 Atlas
    for participant in participants

        cd ${path}${subjects[m]} / label /
        mkdir Destrieux_Atlas
        mkdir DK_Atlas
        mkdir DKT_Atlas

        cd
        Destrieux_Atlas
        mne_annot2labels --subject ${subjects[m]} --parc aparc.a2009s

        cd ../DK_Atlas
        mne_annot2labels --subject ${subjects[m]} --parc aparc

        cd ../DKT_Atlas  # this is the best one to use (at the moment - from the mind-boggle dataset)
        mne_annot2labels --subject ${subjects[m]} --parc aparc.DKTatlas40

        # mne_annot2labels does not seem to be available anymore

    # export to .stl file format, to offer it to participant for 3d printing (if requested)
    for participant in participants
        $ mkdir $SUBJECTS_DIR/participant_01/surf/stl_export_for_3d_printing
        $ mris_convert $SUBJECTS_DIR/participant_01/surf/rh.pial $SUBJECTS_DIR/participant_01/surf/stl_export_for_3d_printing/rh.pial.stl
        $ mris_convert $SUBJECTS_DIR/participant_01/surf/lh.pial $SUBJECTS_DIR/participant_01/surf/stl_export_for_3d_printing/lh.pial.stl
        $ zip $SUBJECTS_DIR/participant_01/surf/stl_export_for_3d_printing/participant_01_stls.zip -q -9 $SUBJECTS_DIR/participant_01/surf/stl_export_for_3d_printing/lh.pial.stl $SUBJECTS_DIR/participant_01/surf/stl_export_for_3d_printing/rh.pial.stl

    #<------------------------------------------------------------->
    """

    import ipdb; ipdb.set_trace()

    # visualise the labels on the pial surface
    # for participant in list_of_participants:
    #    Brain = mne.viz.get_brain_class() # get correct brain class - why is it not doing this automatically?
    #    brain = Brain(participant, hemi="lh", surf="pial", subjects_dir=mri_structurals_directory, size=(800, 600))
    #    brain.add_annotation("aparc.a2009s", borders=False)

    # Computing the 'BEM' surfaces (needed for coregistration to work)
    for participant in list_of_participants:
        #        # andy is using:
        #        https://imaging.mrc-cbu.cam.ac.uk/meg/AnalyzingData/MNE_MRI_processing
        #
        if mri_structural_type == 'T1':
            mne.bem.make_watershed_bem(  # for T1; for FLASH, use make_flash_bem instead
                subject=participant,
                subjects_dir=mri_structurals_directory,
                copy=True,
                overwrite=True,
                show=True,
            )

            mne.bem.make_scalp_surfaces(
                subject=participant,
                subjects_dir=mri_structurals_directory,
                no_decimate=True,
                force=True,
                overwrite=True,
            )

        elif mri_structural_type == 'Flash':
            # mne.bem.make_flash_bem().
            print("Flash not yet implemented.")

        # produce the source space (downsampled version of the cortical surface in Freesurfer), which
        # will be saved in a file ending in *-src.fif
        src = mne.setup_source_space(
            participant,
            spacing="ico5",
            add_dist=True,
            subjects_dir=mri_structurals_directory,
        )
        mne.write_source_spaces(
            Path(
                interim_preprocessing_directory_name,
                "4_hexel_current_reconstruction",
                "src_files",
                participant + "_ico5-src.fif",
            ),
            src,
        )

        # fig = mne.viz.plot_bem(
        #     subject=participant,
        #     subjects_dir=mri_structurals_directory,
        #     brain_surfaces=["white"],
        #     orientation="coronal",
        #     slices=[50, 100, 150, 200],
        # )
        # fig.savefig(
        #     Path(
        #         interim_preprocessing_directory_name,
        #         "4_hexel_current_reconstruction",
        #         "bem_checks",
        #         participant + "_bem_sliced.png",
        #     )
        # )

    # import ipdb; ipdb.set_trace()

    # co-register data (make sure the MEG and EEG is aligned to the head)
    # this will save a trans .fif file
    # for participant in list_of_participants:
    #    mne.gui.coregistration(subject=participant, subjects_dir=mri_structurals_directory, block=True)

    ### #Computing the actual BEM solution
    for participant in list_of_participants:
        # conductivity = (0.3)  # for single layer
        conductivity = (0.3, 0.006, 0.3)  # for three layers
        # Note that only ico=4 is required here: Https://mail.nmr.mgh.harvard.edu/pipermail//mne_analysis/2012-June/001102.html
        model = mne.make_bem_model(
            subject=participant,
            ico=4,
            conductivity=conductivity,
            subjects_dir=mri_structurals_directory,
        )
        bem_sol = mne.make_bem_solution(model)
        output_filename = participant + "-5120-5120-5120-bem-sol.fif"
        mne.bem.write_bem_solution(
            Path(mri_structurals_directory, participant, "bem", output_filename),
            bem_sol,
            overwrite=True,
        )


def create_forward_model_and_inverse_solution(data_root_dir, config: dict):
    list_of_participants = config["participants"]
    dataset_directory_name = config["dataset_directory_name"]

    mri_structurals_directory = Path(
        data_root_dir, dataset_directory_name, config["mri_structurals_directory"]
    )

    interim_preprocessing_directory = Path(
        data_root_dir, dataset_directory_name, "interim_preprocessing_files"
    )

    hexel_current_reconstruction_dir = Path(
        interim_preprocessing_directory, "4_hexel_current_reconstruction"
    )
    hexel_current_reconstruction_dir.mkdir(exist_ok=True)

    coregistration_dir = Path(hexel_current_reconstruction_dir, "coregistration_files")
    coregistration_dir.mkdir(exist_ok=True)

    src_dir = Path(hexel_current_reconstruction_dir, "src_files")
    src_dir.mkdir(exist_ok=True)

    forward_sol_dir = Path(hexel_current_reconstruction_dir, "forward_sol_files")
    forward_sol_dir.mkdir(exist_ok=True)

    inverse_operator_dir = Path(hexel_current_reconstruction_dir, "inverse-operators")
    inverse_operator_dir.mkdir(exist_ok=True)

    # Compute forward solution
    for participant in list_of_participants:
        fwd = mne.make_forward_solution(
            Path(
                data_root_dir,
                dataset_directory_name,
                "raw_emeg",
                participant,
                participant + "_run1_raw.fif",
            ),  # note this file is only used for the sensor positions.
            trans=Path(coregistration_dir, participant + "-trans.fif"),
            src=Path(src_dir, participant + "_ico5-src.fif"),
            bem=Path(
                mri_structurals_directory,
                participant,
                "bem",
                participant + "-5120-5120-5120-bem-sol.fif",
            ),
            meg=config["meg"],
            eeg=config["eeg"],
            mindist=5.0,
            n_jobs=None,
            verbose=True,
        )
        print(fwd)

        # restrict forward vertices to those that make up cortex (i.e. all vertices that are in the annot file,
        # which does not include those in the medial wall)
        labels = mne.read_labels_from_annot(
            subject=participant,
            subjects_dir=mri_structurals_directory,
            parc="aparc.a2009s",
            hemi="both",
        )
        fwd = mne.forward.restrict_forward_to_label(fwd, labels)

        if config["meg"] and config["eeg"]:
            mne.write_forward_solution(
                Path(forward_sol_dir, participant + "-fwd.fif"), fwd=fwd, overwrite=True
            )
        elif config["meg"]:
            mne.write_forward_solution(
                Path(forward_sol_dir, participant + "-megonly-fwd.fif"), fwd=fwd
            )
        elif config["eeg"]:
            mne.write_forward_solution(
                Path(forward_sol_dir, participant + "-eegonly-fwd.fif"), fwd=fwd
            )
        else:
            raise Exception(
                "eeg and meg in the dataset_config file cannot be both False"
            )

    # Compute inverse operator

    for participant in list_of_participants:
        # Read forward solution
        if config["meg"] and config["eeg"]:
            fwd = mne.read_forward_solution(
                Path(forward_sol_dir, participant + "-fwd.fif")
            )
        elif config["meg"]:
            fwd = mne.read_forward_solution(
                Path(forward_sol_dir, participant + "-megonly-fwd.fif")
            )
        elif config["eeg"]:
            fwd = mne.read_forward_solution(
                Path(forward_sol_dir, participant + "-eegonly-fwd.fif")
            )

        # Read noise covariance matrix
        if config["duration"] is None or config["cov_method"] != "emptyroom":
            noise_cov = mne.read_cov(
                str(
                    Path(
                        hexel_current_reconstruction_dir,
                        "noise_covariance_files",
                        participant + "-" + config["cov_method"] + "-cov.fif",
                    )
                )
            )
        else:
            noise_cov = mne.read_cov(
                str(
                    Path(
                        hexel_current_reconstruction_dir,
                        "noise_covariance_files",
                        participant
                        + "-"
                        + config["cov_method"]
                        + str(config["duration"])
                        + "-cov.fif",
                    )
                )
            )

        # note this file is only used for the sensor positions.
        raw = mne.io.Raw(
            Path(
                Path(path.abspath("")),
                interim_preprocessing_directory,
                "2_cleaned",
                participant + "_run1_cleaned_raw.fif.gz",
            )
        )

        inverse_operator = mne.minimum_norm.make_inverse_operator(
            raw.info,  # note this file is only used for the sensor positions.
            fwd,
            noise_cov,
            loose=0.2,
            depth=None,
            use_cps=True,
        )
        if config["meg"] and config["eeg"]:
            mne.minimum_norm.write_inverse_operator(
                str(
                    Path(
                        inverse_operator_dir,
                        participant
                        + "_ico5-3L-loose02-cps-nodepth-"
                        + config["cov_method"]
                        + "-inv.fif",
                    )
                ),
                inverse_operator,
                overwrite=True,
            )
        elif config["meg"]:
            if config["duration"] is None:
                mne.minimum_norm.write_inverse_operator(
                    str(
                        Path(
                            inverse_operator_dir,
                            participant
                            + "_ico5-3L-loose02-cps-nodepth-megonly-"
                            + config["cov_method"]
                            + "-inv.fif",
                        )
                    ),
                    inverse_operator,
                )
            else:
                mne.minimum_norm.write_inverse_operator(
                    str(
                        Path(
                            inverse_operator_dir,
                            participant
                            + "_ico5-3L-loose02-cps-nodepth-megonly-"
                            + config["cov_method"]
                            + str(config["duration"])
                            + "-inv.fif",
                        )
                    ),
                    inverse_operator,
                )
        elif config["eeg"]:
            mne.minimum_norm.write_inverse_operator(
                str(
                    Path(
                        inverse_operator_dir,
                        participant
                        + "_ico5-3L-loose02-cps-nodepth-eegonly-"
                        + config["cov_method"]
                        + "-inv.fif",
                    )
                ),
                inverse_operator,
            )


def confirm_digitisation_locations(data_root_dir, config):
    list_of_participants = config["participants"]

    dataset_directory_name = config["dataset_directory_name"]
    interim_preprocessing_directory = Path(
        data_root_dir, dataset_directory_name, "interim_preprocessing_files"
    )
    hexel_current_reconstruction_dir = Path(
        interim_preprocessing_directory, "4_hexel_current_reconstruction"
    )
    mri_structurals_directory = Path(
        data_root_dir, dataset_directory_name, config["mri_structurals_directory"]
    )
    coregistration_dir = Path(hexel_current_reconstruction_dir, "coregistration_files")
    src_dir = Path(hexel_current_reconstruction_dir, "src_files")
    fwd_dir = Path(hexel_current_reconstruction_dir, "forward_sol_files")
    coregistration_checks_dir = Path(
        coregistration_dir, "confirmation_of_digitisation_locations"
    )
    helmet_intersection_check_dir = Path(
        coregistration_checks_dir, "helmet-intersection-check"
    )
    sensor_location_check_dir = Path(coregistration_checks_dir, "sensor-location-check")
    src_check_dir = Path(coregistration_checks_dir, "src-check")
    fwd_check_dir = Path(coregistration_checks_dir, "fwd-check")
    medial_wall_omitted_check_dir = Path(
        coregistration_checks_dir, "medial-wall-omitted-check"
    )
    side_check_dir = Path(coregistration_checks_dir, "side-check")

    coregistration_checks_dir.mkdir(exist_ok=True)
    helmet_intersection_check_dir.mkdir(exist_ok=True)
    sensor_location_check_dir.mkdir(exist_ok=True)
    src_check_dir.mkdir(exist_ok=True)
    fwd_check_dir.mkdir(exist_ok=True)
    medial_wall_omitted_check_dir.mkdir(exist_ok=True)
    side_check_dir.mkdir(exist_ok=True)

    for participant in list_of_participants:
        raw_fname = Path(
            interim_preprocessing_directory,
            "2_cleaned",
            participant + "_run1_cleaned_raw.fif.gz",
        )
        trans_fname = Path(coregistration_dir, participant + "-trans.fif")
        src_name = Path(src_dir, participant + "_ico5-src.fif")
        if config["meg"] and config["eeg"]:
            fwd_name = Path(fwd_dir, participant + "-fwd.fif")
        elif config["meg"] and not config["eeg"]:
            fwd_name = Path(fwd_dir, participant + "-megonly-fwd.fif")
        raw = mne.io.read_raw_fif(raw_fname)
        trans = mne.read_trans(trans_fname)
        src = mne.read_source_spaces(src_name)
        fwd = mne.read_forward_solution(fwd_name)

        fwd_fixed = mne.convert_forward_solution(
            fwd, surf_ori=True, force_fixed=True, use_cps=True
        )

        # Load the T1 file and change the header information to the correct units
        t1w = nib.load(Path(mri_structurals_directory, participant, "mri", "T1.mgz"))
        t1w = nib.Nifti1Image(t1w.dataobj, t1w.affine)
        t1w.header["xyzt_units"] = np.array(10, dtype="uint8")

        fig = mne.viz.plot_alignment(
            raw.info,
            trans=trans,
            subject=participant,
            subjects_dir=mri_structurals_directory,
            surfaces=["head-dense", "white"],
            show_axes=True,
            dig=True,
            meg={"helmet": 0.1, "sensors": 0.1},
            coord_frame="meg",
            mri_fiducials="estimated",
        )
        mne.viz.set_3d_view(fig, 0, 90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))
        fig.plotter.screenshot(
            Path(
                helmet_intersection_check_dir,
                participant + "-helmet-intersection-check.png",
            )
        )

        fig = mne.viz.plot_alignment(
            raw.info,
            trans=trans,
            subject=participant,
            subjects_dir=mri_structurals_directory,
            surfaces=["head-dense", "white"],
            show_axes=True,
            dig=True,
            eeg=dict(original=0, projected=1),
            meg={"sensors": 0.1},
            coord_frame="meg",
            mri_fiducials="estimated",
        )
        mne.viz.set_3d_view(fig, 45, 90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))
        fig.plotter.screenshot(
            Path(sensor_location_check_dir, participant + "-sensor-location-check.png")
        )

        fig = mne.viz.plot_alignment(
            raw.info,
            trans=trans,
            subject=participant,
            src=src,
            subjects_dir=mri_structurals_directory,
            surfaces=["white"],
            eeg=False,
            dig="fiducials",
            show_axes=True,
            meg={"sensors": 0.0},
            coord_frame="meg",
        )
        mne.viz.set_3d_view(fig, 45, 90, distance=0.4, focalpoint=(0.0, 0.0, 0.0))
        fig.plotter.screenshot(Path(src_check_dir, participant + "-src-check.png"))

        fig = mne.viz.plot_alignment(
            raw.info,
            trans=trans,
            subject=participant,
            dig=False,
            eeg=False,
            fwd=fwd_fixed,
            subjects_dir=mri_structurals_directory,
            surfaces=["white"],
            show_axes=True,
            meg={"sensors": 0.0},
            coord_frame="meg",
        )
        mne.viz.set_3d_view(fig, 45, 90, distance=0.4, focalpoint=(0.0, 0.0, 0.0))
        fig.plotter.screenshot(Path(fwd_check_dir, participant + "-fwd-check.png"))

        fig = mne.viz.plot_alignment(
            raw.info,
            trans=trans,
            subject=participant,
            dig=False,
            eeg=False,
            fwd=fwd_fixed,
            subjects_dir=mri_structurals_directory,
            surfaces=["white"],
            show_axes=True,
            meg={"sensors": 0.0},
            coord_frame="meg",
        )
        mne.viz.set_3d_view(fig, 0, 180, distance=0.4, focalpoint=(0.0, 0.0, 0.0))
        fig.plotter.screenshot(
            Path(
                medial_wall_omitted_check_dir,
                participant + "-medial-wall-omitted-check.png",
            )
        )

        fig = mne.viz.plot_alignment(
            raw.info,
            trans=trans,
            subject=participant,
            dig=False,
            eeg=False,
            fwd=fwd_fixed,
            subjects_dir=mri_structurals_directory,
            surfaces=["white"],
            show_axes=True,
            meg={"sensors": 0.0},
            coord_frame="meg",
        )
        mne.viz.set_3d_view(fig, 0, 90, distance=0.4, focalpoint=(0.0, 0.0, 0.0))
        fig.plotter.screenshot(Path(side_check_dir, participant + "-side-check.png"))


def create_hexel_morph_maps(data_root_dir, config: dict):
    list_of_participants = config["participants"]
    dataset_directory_name = config["dataset_directory_name"]

    mri_structurals_directory = Path(
        data_root_dir, dataset_directory_name, config["mri_structurals_directory"]
    )

    interim_preprocessing_directory = Path(
        data_root_dir, dataset_directory_name, "interim_preprocessing_files"
    )

    hexel_current_reconstruction_dir = Path(
        interim_preprocessing_directory, "4_hexel_current_reconstruction"
    )
    hexel_current_reconstruction_dir.mkdir(exist_ok=True)

    src_dir = Path(hexel_current_reconstruction_dir, "src_files")
    src_dir.mkdir(exist_ok=True)

    forward_sol_dir = Path(hexel_current_reconstruction_dir, "forward_sol_files")
    forward_sol_dir.mkdir(exist_ok=True)

    morph_map_dir = Path(hexel_current_reconstruction_dir, "morph_maps")
    morph_map_dir.mkdir(exist_ok=True)

    for participant in list_of_participants:
        morphmap_filename = Path(morph_map_dir, participant + "_fsaverage_morph.h5")

        # First compute morph matrices for participant
        if not path.isfile(morphmap_filename):
            # read the src space not from the original but from the version in fwd or
            # inv, incase any vertices have been removed due to proximity to the scalp
            # https://mne.tools/stable/auto_tutorials/forward/30_forward.html#sphx-glr-auto-tutorials-forward-30-forward-py
            if config["meg"] and config["eeg"]:
                fwd = mne.read_forward_solution(
                    Path(forward_sol_dir, participant + "-fwd.fif")
                )
            elif config["meg"] and not config["eeg"]:
                fwd = mne.read_forward_solution(
                    Path(forward_sol_dir, participant + "-megonly-fwd.fif")
                )
            src_from = fwd["src"]

            src_to = mne.read_source_spaces(Path(src_dir, "fsaverage_ico5-src.fif"))

            morph = mne.compute_source_morph(
                src_from,
                subject_from=participant,
                subject_to="fsaverage",
                src_to=src_to,
                subjects_dir=mri_structurals_directory,
            )
            if isinstance(morph.morph_mat, scipy.sparse._csr.csr_array):
                morph.morph_mat = csr_matrix(morph.morph_mat)
            morph.save(morphmap_filename)
        else:
            _logger.info("Morph maps already created")

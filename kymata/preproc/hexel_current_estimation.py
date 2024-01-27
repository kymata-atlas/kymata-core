from os import path
from pathlib import Path

import mne


def create_current_estimation_prerequisites(data_root_dir, config: dict):
    """
    Copy the structurals to the local Kymata folder,
    create the surfaces, the boundary element model solutions, and the source space
    """

    list_of_participants = config['list_of_participants']
    dataset_directory_name = config['dataset_directory_name']
    intrim_preprocessing_directory_name = Path(data_root_dir, dataset_directory_name, "intrim_preprocessing_files")
    mri_structural_type = config['mri_structural_type'] 
    mri_structurals_directory = Path(data_root_dir, dataset_directory_name, config['mri_structurals_directory'])

    '''    

    # <--------------------Command Line-------------------------->

    # Set location in the Kymata Project directory
    # where the converted MRI structurals will reside, and create the folder structure
    $ freesurfer_6.0.0
    $ setenv SUBJECTS_DIR /imaging/projects/cbu/kymata/data/dataset_4-english-narratives/raw_mri_structurals/
    for all participants:
        $ mksubjdirs participant_01 # note - this appears to ignore SUBJECTS_DIR and uses the folder you are in.

    # Load the fsaverage mesh
    $ cp -r $FREESURFER_HOME/subjects/fsaverage $SUBJECTS_DIR/fsaverage
    
    # create a one-off source space of fsaverage to morph to later.
    src = mne.setup_source_space(
        "fsaverage", spacing="ico5", subjects_dir=mri_structurals_directory, verbose=True
    )
    mne.write_source_spaces(Path(intrim_preprocessing_directory_name,
                                     "4_hexel_current_reconstruction",
                                     "src_files",
                                     "fsaverage_ico5-src.fif"), src)
    
    # todo - we were using the "aparc.DKTatlas40" atlas - is this the same as the Desikan-Killiany Atlas, and aparc.DKTatlas?
    # (?h.aparc.annot)? https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation. And if so, do we
    # need to add it to the fsaverage folder for fsaverage? (I don't think it is used here, although we do
    # use it in Kymata web. If so, remove this section). i.e.
    #$ cp $FREESURFER_HOME/average/rh.DKTatlas40.gcs  $SUBJECTS_DIR/fsaverage/rh.DKTatlas40.gcs
    #$ cp $FREESURFER_HOME/average/lh.DKTatlas40.gcs  $SUBJECTS_DIR/fsaverage/lh.DKTatlas40.gcs
    #$ mris_ca_label -orig white -novar fsaverage rh sphere.reg $SUBJECTS_DIR/fsaverage/label/rh.DKTatlas40.gcs $SUBJECTS_DIR/fsaverage/label/rh.aparc.DKTatlas40.annot
    #$ mris_ca_label -orig white -novar fsaverage lh sphere.reg $SUBJECTS_DIR/fsaverage/label/lh.DKTatlas40.gcs $SUBJECTS_DIR/fsaverage/label/lh.aparc.DKTatlas40.annot

    # move data across from the MRIdata folder to the local
    # directory, so freesurfer can find it - also convert from dcm to .mgz
    for participant in participants
        $ mri_convert /mridata/cbu/CBU230790_MEG23008/20231102_130449/Series005_CBU_MPRAGE_32chn/1.3.12.2.1107.5.2.43.67035.202311021312263809335255.dcm $SUBJECTS_DIR/participant_01/mri/orig/001.mgz

    # creates suitable T1, meshes and labels
    for participant in participants
        $ recon-all -s participant_01 -all

        #todo - I think this does everything at once (folders and ), so might be better if there is a python version in the future
        $ recon-all -i $SUBJECTS_DIR/participant_01/mri/orig/001.mgz -s participant_01 -all

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

        cd ../DKT_Atlas  # this is the best one to use (at the momment - from the mind-boggle dataset)
        mne_annot2labels --subject ${subjects[m]} --parc aparc.DKTatlas40

    # export to .stl file format, to offer it to participant for 3d printing (if requested)
    for participant in participants
        $ mkdir $SUBJECTS_DIR/participant_01/surf/stl_export_for_3d_printing
        $ mris_convert $SUBJECTS_DIR/participant_01/surf/rh.pial $SUBJECTS_DIR/participant_01/surf/stl_export_for_3d_printing/rh.pial.stl
        $ mris_convert $SUBJECTS_DIR/participant_01/surf/lh.pial $SUBJECTS_DIR/participant_01/surf/stl_export_for_3d_printing/lh.pial.stl
        $ zip $SUBJECTS_DIR/participant_01/surf/stl_export_for_3d_printing/participant_01_stls.zip -q -9 $SUBJECTS_DIR/participant_01/surf/stl_export_for_3d_printing/lh.pial.stl $SUBJECTS_DIR/participant_01/surf/stl_export_for_3d_printing/rh.pial.stl

    #<------------------------------------------------------------->
    '''
    # visualise the labels on the pial surface
    # for participant in list_of_participants:
    #    Brain = mne.viz.get_brain_class() # get correct brain class - why is it not doing this automatically?
    #    brain = Brain(participant, hemi="lh", surf="pial", subjects_dir=mri_structurals_directory, size=(800, 600))
    #    brain.add_annotation("aparc.a2009s", borders=False)

    # Computing the 'BEM' surfaces (needed for coregistration to work)
    for participant in list_of_participants:
        #        # andy is using:
        #        https://imaging.mrc-cbu.cam.ac.uk/meg/AnalyzingData/MNE_MRI_processing
        #        # todo -  AT has used the commandline version to create the BEMSs: do and then compare
        #
        #        if mri_structural_type == 'T1':
        #            mne.bem.make_watershed_bem(  # for T1; for FLASH, use make_flash_bem instead
        #                subject=participant,
        #                subjects_dir=mri_structurals_directory,
        #                copy=True,
        #                overwrite=True,
        #                show=True,
        #            )
        #
        #            mne.bem.make_scalp_surfaces(
        #                subject=participant,
        #                subjects_dir=mri_structurals_directory,
        #                no_decimate=True,
        #                force=True,
        #                overwrite=True,
        #            )
        #
        #        elif mri_structural_type == 'Flash':
        #            # todo add & test flash
        #            # mne.bem.make_flash_bem().
        #            print("Flash not yet implemented.")

        # produce the source space (downsampled version of the cortical surface in Freesurfer), which
        # will be saved in a file ending in *-src.fif
        src = mne.setup_source_space(
            participant, spacing="ico5", add_dist=True, subjects_dir=mri_structurals_directory
        )
        mne.write_source_spaces(Path(intrim_preprocessing_directory_name,
                                     "4_hexel_current_reconstruction",
                                     "src_files",
                                     participant + "_ico5-src.fif"), src)

    #        mne.viz.plot_bem(subject=participant,
    #                         subjects_dir=mri_structurals_directory,
    #                         brain_surfaces="white",
    #                         orientation="coronal",
    #                         slices=[50, 100, 150, 200])

    # co-register data (make sure the MEG and EEG is aligned to the head)
    # this will save a trans .fif file
    # for participant in list_of_participants:
    #    mne.gui.coregistration(subject=participant, subjects_dir=mri_structurals_directory, block=True)

    ### #Computing the actual BEM solution
    for participant in list_of_participants:
        # conductivity = (0.3)  # for single layer
        conductivity = (0.3, 0.006, 0.3)  # for three layers
        # Note that only ico=4 is required here: Https://mail.nmr.mgh.harvard.edu/pipermail//mne_analysis/2012-June/001102.html
        model = mne.make_bem_model(subject=participant, ico=4,
                                   conductivity=conductivity,
                                   subjects_dir=mri_structurals_directory)
        bem_sol = mne.make_bem_solution(model)
        output_filename = participant + '-5120-5120-5120-bem-sol.fif'
        mne.bem.write_bem_solution(Path(mri_structurals_directory, participant, 'bem',
                                        output_filename), bem_sol)


def create_forward_model_and_inverse_solution(data_root_dir, config: dict):

    list_of_participants = config['list_of_participants']
    dataset_directory_name = config['dataset_directory_name']
    intrim_preprocessing_directory_name = Path(data_root_dir, dataset_directory_name, "intrim_preprocessing_files")
    mri_structurals_directory = Path(data_root_dir, dataset_directory_name, config['mri_structurals_directory'])

    # Compute forward solution
    for participant in list_of_participants:
         fwd = mne.make_forward_solution(
             # Path(Path(path.abspath("")), "data",
             Path(data_root_dir,
                  dataset_directory_name,
                  'raw_emeg', participant, participant +
                  '_run1_raw.fif'), # note this file is only used for the sensor positions.
             trans=Path(intrim_preprocessing_directory_name, "4_hexel_current_reconstruction","coregistration_files", participant + '-trans.fif'),
             src=Path(intrim_preprocessing_directory_name, "4_hexel_current_reconstruction","src_files", participant + '_ico5-src.fif'),
             bem=Path(mri_structurals_directory, participant, "bem", participant + '-5120-5120-5120-bem-sol.fif'),
             meg=config['meg'],
             eeg=config['eeg'],
             mindist=5.0,
             n_jobs=None,
             verbose=True,
         )
         print(fwd)
         if config['meg'] and config['eeg']:
             mne.write_forward_solution(Path(intrim_preprocessing_directory_name, "4_hexel_current_reconstruction","forward_sol_files", participant + '-fwd.fif'), fwd)
         elif config['meg']:
             mne.write_forward_solution(Path(intrim_preprocessing_directory_name, "4_hexel_current_reconstruction","forward_sol_files", participant + '-fwd-megonly.fif'), fwd)
         elif config['eeg']:
             mne.write_forward_solution(Path(intrim_preprocessing_directory_name, "4_hexel_current_reconstruction","forward_sol_files", participant + '-fwd-eegonly.fif'), fwd)
         else:
             raise Exception('eeg and meg in the config file cannot be both False')

    # Compute inverse operator

    for participant in list_of_participants:

        # Read forward solution
        if config['meg'] and config['eeg']:
            fwd = mne.read_forward_solution(Path(
                intrim_preprocessing_directory_name,
                "4_hexel_current_reconstruction",
                "forward_sol_files",
                participant + '-fwd.fif'))
        elif config['meg']:
            fwd = mne.read_forward_solution(Path(
                intrim_preprocessing_directory_name,
                "4_hexel_current_reconstruction",
                "forward_sol_files",
                participant + '-fwd-megonly.fif'))
        elif config['eeg']:
            fwd = mne.read_forward_solution(Path(
                intrim_preprocessing_directory_name,
                "4_hexel_current_reconstruction",
                "forward_sol_files",
                participant + '-fwd-eegonly.fif'))
            
        # Read noise covariance matrix
        if config['duration'] == None:
            noise_cov = mne.read_cov(str(Path(
                intrim_preprocessing_directory_name,
                '3_evoked_sensor_data',
                'covariance_grand_average',
                participant + '-auto-cov-' + config['cov_method'] + '.fif')))
        else:
            noise_cov = mne.read_cov(str(Path(
            intrim_preprocessing_directory_name,
                '3_evoked_sensor_data',
                'covariance_grand_average',
                participant + '-auto-cov-' + config['cov_method'] + str(config['duration']) + '.fif')))
        
        # note this file is only used for the sensor positions.
        raw = mne.io.Raw(Path(
            Path(path.abspath("")),
            intrim_preprocessing_directory_name,
            '2_cleaned',
            participant + '_run1_cleaned_raw.fif.gz'))

        inverse_operator = mne.minimum_norm.make_inverse_operator(
            raw.info,  # note this file is only used for the sensor positions.
            fwd,
            noise_cov,
            loose=0.2,
            depth=None,
            use_cps=True
        )
        if config['meg'] and config['eeg']:
            mne.minimum_norm.write_inverse_operator(
                str(Path(
                    intrim_preprocessing_directory_name,
                    '4_hexel_current_reconstruction',
                    'inverse-operators',
                    participant + '_ico5-3L-loose02-cps-nodepth-' + config['cov_method'] + '-inv.fif')), 
                inverse_operator)
        elif config['meg']:
            if config['duration'] == None:
                mne.minimum_norm.write_inverse_operator(
                    str(Path(
                        intrim_preprocessing_directory_name,
                        '4_hexel_current_reconstruction',
                        'inverse-operators',
                        participant + '_ico5-3L-loose02-cps-nodepth-megonly-' + config['cov_method'] + '-inv.fif')), 
                    inverse_operator)
            else:
                mne.minimum_norm.write_inverse_operator(
                    str(Path(
                        intrim_preprocessing_directory_name,
                        '4_hexel_current_reconstruction',
                        'inverse-operators',
                        participant + '_ico5-3L-loose02-cps-nodepth-megonly-' + config['cov_method'] + str(config['duration']) + '-inv.fif')), 
                    inverse_operator)               
        elif config['eeg']:
            mne.minimum_norm.write_inverse_operator(
                str(Path(
                    intrim_preprocessing_directory_name,
                    '4_hexel_current_reconstruction',
                    'inverse-operators',
                    participant + '_ico5-3L-loose02-cps-nodepth-eegonly-' + config['cov_method'] + '-inv.fif')), 

                inverse_operator)


def create_hexel_current_files(data_root_dir, config: dict):

    number_of_trials = config['number_of_trials']
    list_of_participants = config['list_of_participants']
    dataset_directory_name = config['dataset_directory_name']
    intrim_preprocessing_directory_name = Path(data_root_dir, dataset_directory_name, "intrim_preprocessing_files")
    mri_structurals_directory = Path(data_root_dir, dataset_directory_name, config['mri_structurals_directory'])
    input_streams = config['input_streams']

    snr = 1 # default is 3
    lambda2 = 1.0 / snr ** 2

    for participant in list_of_participants:

        morphmap_filename = Path(intrim_preprocessing_directory_name,
                                 "4_hexel_current_reconstruction",
                                 "morph_maps",
                                 participant + "_fsaverage_morph.h5")

        # First compute morph matrices for participant
        if not path.isfile(morphmap_filename):
            
            # read the src space not from the original but from the version in fwd or
            # inv, incase an vertices have been removed due to proximity to the scalp
            # https://mne.tools/stable/auto_tutorials/forward/30_forward.html#sphx-glr-auto-tutorials-forward-30-forward-py
            fwd = mne.read_forward_solution(Path(
                intrim_preprocessing_directory_name,
                "4_hexel_current_reconstruction",
                "forward_sol_files",
                participant + '-fwd.fif'))
            src_from = fwd['src']
            
            src_to = mne.read_source_spaces(Path(
                intrim_preprocessing_directory_name,
                "4_hexel_current_reconstruction",
                "src_files",
                'fsaverage_ico5-src.fif'))

            morph = mne.compute_source_morph(
                src_from,
                subject_from=participant,
                subject_to="fsaverage",
                src_to=src_to,
                subjects_dir=mri_structurals_directory,
            )
            morph.save(morphmap_filename)
        else:
            morph = mne.read_source_morph(morphmap_filename)

        # Compute source stcs
        inverse_operator = mne.minimum_norm.read_inverse_operator(str(Path(
            intrim_preprocessing_directory_name,
            '4_hexel_current_reconstruction',
            'inverse-operators',
            participant + '_ico5-3L-loose02-cps-nodepth-inv.fif')))

        for input_stream in input_streams:
            for trial in range(1,number_of_trials+1):
                # Apply Inverse
                evoked = mne.read_evokeds(str(Path(
                        intrim_preprocessing_directory_name,
                        '3_evoked_sensor_data',
                        'evoked_data',
                        input_stream,
                        participant + '_item' + str(trial) + '-ave.fif')),
                    condition=0, baseline=None)
                mne.set_eeg_reference(evoked, projection=True)
                stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2, "MNE", pick_ori='normal')

                # Morph to average
                stc_morphed_to_fsaverage = morph.apply(stc)
                stc_morphed_to_fsaverage.save(str(Path(
                    intrim_preprocessing_directory_name,
                    '5-single-trial-source-data',
                    'vert10242-nodepth-diagonly-snr1-signed-fsaverage-baselineNone',
                    input_stream,
                    participant + '-' + str(trial))))


def average_participants_hexel_currents(participants, inputstream):
    from functools import reduce

    dataset_root = Path('/imaging/at03/NKG_Data_Sets/DATASET_3-01_visual-and-auditory/')

    with Path(dataset_root, 'items.txt').open("r") as f:
        words = list(f.read().split())

    stc_dir = Path(dataset_root,
                   '4-single-trial-source-data',
                   'vert10242-nodepth-diagonly-snr1-signed-fsaverage-baselineNone',
                   inputstream)

    for w in words:
        stcs = [
            mne.read_source_estimate(
                str(Path(stc_dir, f"{subject}{w}-lh.stc")),
                subject='fsaverage')
            for subject in participants
        ]

        # take mean average
        stc_avg = reduce(lambda x, y: x + y, stcs)
        stc_avg /= len(stcs)
        stc_avg.save(str(Path(
            dataset_root,
            '5-averaged-by-trial-data',
            'vert10242-nodepth-diagonly-snr1-signed-fsaverage-baselineNone',
            inputstream,
            w)))

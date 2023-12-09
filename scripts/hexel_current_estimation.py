import mne
from os import path
from pathlib import Path

def create_current_estimation_prerequisites(config: dict):
    '''
    Copy the structurals to the local Kymata folder,
    create the surfaces, the boundary element model solutions, and the source space
    '''

    list_of_participants = config['list_of_participants']
    dataset_directory_name = config['dataset_directory_name']
    intrim_preprocessing_directory_name = Path(Path(path.abspath("")), "data", dataset_directory_name,
                                               "intrim_preprocessing_files")
    mri_structural_type = config['mri_structural_type']
    mri_structurals_directory = config['mri_structurals_directory']
    mri_structurals_directory = Path(Path(path.abspath("")), "data", dataset_directory_name, mri_structurals_directory)

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

    #<------------------------------------------------------------->
'''
    # visualise the labels on the pial surface
    #for participant in list_of_participants:
    #    Brain = mne.viz.get_brain_class() # get correct brain class - why is it not doing this automatically?
    #    brain = Brain(participant, hemi="lh", surf="pial", subjects_dir=mri_structurals_directory, size=(800, 600))
    #    brain.add_annotation("aparc.a2009s", borders=False)

    # Computing the 'BEM' surfaces (needed for coregistration to work)
    for  participant in list_of_participants:
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
    #for participant in list_of_participants:
    #    mne.gui.coregistration(subject=participant, subjects_dir=mri_structurals_directory, block=True)

### #Computing the actual BEM solution
    for participant in list_of_participants:
        #conductivity = (0.3)  # for single layer
        conductivity = (0.3, 0.006, 0.3)  # for three layers
        #Note that only ico=4 is required here: Https://mail.nmr.mgh.harvard.edu/pipermail//mne_analysis/2012-June/001102.html
        model = mne.make_bem_model(subject=participant, ico=4,
                                   conductivity=conductivity,
                                   subjects_dir=mri_structurals_directory)
        bem_sol = mne.make_bem_solution(model)
        output_filename = participant + '-5120-5120-5120-bem-sol.fif'
        mne.bem.write_bem_solution(Path(mri_structurals_directory, participant, 'bem',
                                   output_filename), bem_sol)


def create_forward_model_and_inverse_solution(config: dict):

    list_of_participants = config['list_of_participants']
    dataset_directory_name = config['dataset_directory_name']
    intrim_preprocessing_directory_name = Path(Path(path.abspath("")), "data", dataset_directory_name, "intrim_preprocessing_files")
    mri_structurals_directory = config['mri_structurals_directory']
    mri_structurals_directory = Path(Path(path.abspath("")), "data", dataset_directory_name, mri_structurals_directory)

    # Compute forward solution
    for participant in list_of_participants:
        fwd = mne.make_forward_solution(
            Path(Path(path.abspath("")), "data",
                 dataset_directory_name,
                 'raw', participant, participant +
                 '_run1_raw.fif'), # note this file is only used for the sensor positions.
            trans=Path(intrim_preprocessing_directory_name, "4_hexel_current_reconstruction","coregistration_files", participant + '-trans.fif'),
            src=Path(intrim_preprocessing_directory_name, "4_hexel_current_reconstruction","src_files", participant + '_ico5-src.fif'),
            bem=Path(mri_structurals_directory, participant, "bem", participant + '-5120-5120-5120-bem-sol.fif'),
            meg=True,
            eeg=True,
            mindist=5.0,
            n_jobs=None,
            verbose=True,
        )
        print(fwd)
        mne.write_forward_solution(Path(intrim_preprocessing_directory_name, "4_hexel_current_reconstruction","forward_sol_files", participant + '-fwd.fif'), fwd)

    '''
    # Compute inverse operator
    for participant in list_of_participants:

        http://martinos.org/mne/stable/auto_examples/inverse/plot_make_inverse_operator.html#sphx-glr-auto-examples-inverse-plot-make-inverse-operator-py

        ... BE SURE TO copy exactly what was in the origional create operator script

        #    # Apply maxwell filtering (and everything else, such as filtering) to the empty room
        #
        #    .maxwell_filter_prepare_emptyroom,
        #    .maxwell_filter

        # USe Empty room Max filtered for covarience!
        # USe em[ty room for MEG and dia for EEG etc] (is on MNE)

def create_hexel_current_files():

    snr = 1 # default is 3
    lambda2 = 1.0 / snr ** 2

    for p in participants:

        # First compute morph matrices for participant
        src_to = mne.read_source_spaces(fname_fsaverage_src)
        print(src_to[0]["vertno"])  # special, np.arange(10242)
        morph = mne.compute_source_morph(
            stc,
            subject_from="sample",
            subject_to="fsaverage",
            src_to=src_to,
            subjects_dir=subjects_dir,
        )

        # Compute source stcs
        inverse_operator = read_inverse_operator((data_path + '3-sensor-data/inverse-operators/' + p + '_ico-53L-loose02-diagnoise-nodepth-reg-inv-csd.fif'))

        for w in words:
            # Apply Inverse
            evoked = read_evokeds((data_path + '3-sensor-data/fif-out/' + inputstream + '/' + p + '_' + w + '-ave.fif'),
                                  condition=0, baseline=None)

            if (evoked.nave > 0):
                stc_from = apply_inverse(evoked, inverse_operator, lambda2, "MNE", pick_ori='normal')

                # Morph to average
                stc_from.subject = subject_from  # only needed if subject has been tested previously and so has a different subject number
                stc_morphed = mne.morph_data_precomputed(subject_from, subject_to, stc_from, vertices_to, morph_mat)
                stc_morphed.save((data_path + '/4-single-trial-source-data/vert10242-nodepth-diagonly-snr1-signed-fsaverage-baselineNone/' + inputstream + '/' + p + '-' + w))


def average_participants_hexel_currents():

    f = open('/imaging/at03/NKG_Data_Sets/DATASET_3-01_visual-and-auditory/items.txt', 'r')
    words = list(f.read().split())

    stcdir = '/imaging/at03/NKG_Code_output/Version5/DATASET_3-01_visual-and-auditory/4-single-trial-source-data/vert10242-nodepth-diagonly-snr1-signed-fsaverage-baselineNone/' + inputstream + '/'

    for w in words:
        fname = os.path.join(stcdir, '%s-' + w + '-lh.stc')
        stcs = [mne.read_source_estimate(fname % subject, subject='fsaverage') for subject in participants]

        # take mean average
        stc_avg = reduce(lambda x, y: x + y, stcs)
        stc_avg /= len(stcs)
        stc_avg.save('/imaging/at03/NKG_Code_output/Version5/DATASET_3-01_visual-and-auditory/5-averaged-by-trial-data/vert10242-nodepth-diagonly-snr1-signed-fsaverage-baselineNone/' + inputstream + '/' + w))
'''
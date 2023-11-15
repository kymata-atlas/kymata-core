import mne
import surfa

def create_current_estimation_prerequisites():

    # Create the boudary element model, the forward model and inverse solution

    set location in the Kymata Project directory where the converted MRI structurals will reside
    load in fsaverage mesh # this is the mesh we will use
    load in fsaverage labels # this is the labels we will use, the aparc.DKTatlas40


    # move data across from the MRIdata folder to the local directory, so freesurfer can find it
    for participant in participants
        mri_convert < / mridata / cbu / * / * / onedcmfile.dcm > < / myMRIdirectory / mysubjectname / mri / orig / 001.
        mgz>

    # create strucruals meshes and labels
    for participant in participants
        $recon-all -subjid <mysubjectname> -autorecon1
        Do i need this, can i not use: ???????? DO I NEED TO DO THIS?
        https: // mne.tools / stable / generated / mne.bem.make_scalp_surfaces.html
        or use surfa package

    # create labels for these avergae, for kymata we prefer aparc.DKTatlas40
    for participant in participants
        # my_subject=sample
        # my_NIfTI=/path/to/NIfTI.nii.gz
        # recon-all -i $my_NIfTI -s $my_subject participant_1

    # visualise the labels on the pial surface
    for participant in participants
        Brain = mne.viz.get_brain_class()
        brain = Brain(
            "sample", hemi="lh", surf="pial", subjects_dir=subjects_dir, size=(800, 600)
        )
        brain.add_annotation("aparc.a2009s", borders=False)

    # co-register data (make sure the MEG and EEG is alligned to the head)
    # this will save a trans .fif file
    for  participant in participants
        mne.gui.coregistration(subject="sample", subjects_dir=subjects_dir, output= XXXX)

    # Computing the BEM Surfaces
    for  participant in participants
        if normal mre
            mne.bem.make_watershed_bem

            mne.viz.plot_bem(subject=subject,
                subjects_dir=subjects_dir,
                brain_surfaces="white",
                orientation="coronal",
                slices=[50, 100, 150, 200],)

        else if Flash
            mne.bem.make_flash_bem().

    #Check eeg is in correct place (can be merged with next one?)
    for  participant in participants
        # Plot electrode locations on scalp
        fig = plot_alignment(
            raw.info,
            trans,
            subject="sample",
            dig=False,
            eeg=["original", "projected"],
            meg=[],
            coord_frame="head",
            subjects_dir=subjects_dir,
        )

        # Set viewing angle
        set_3d_view(figure=fig, azimuth=135, elevation=80)

    #Set up the source space
    for  participant in participants
        src = mne.setup_source_space(
            subject, spacing="oct4", add_dist="patch", subjects_dir=subjects_dir
        )
        print(src)
        mne.viz.plot_bem(src=src, **plot_bem_kwargs)

        fig = mne.viz.plot_alignment(
            subject=subject,
            subjects_dir=subjects_dir,
            surfaces="white",
            coord_frame="mri",
            src=src,
        )
        mne.viz.set_3d_view(
            fig,
            azimuth=173.78,
            elevation=101.75,
            distance=0.30,
            focalpoint=(-0.03, -0.01, 0.03),
        )

    #Computing the actual BEM solution
    for participant in participants
        conductivity = (0.3,)  # for single layer
        # conductivity = (0.3, 0.006, 0.3)  # for three layers
        model = mne.make_bem_model(subject='sample', ico=4,
                                   conductivity=conductivity,
                                   subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)
        mne.bem.write_bem_solution(subjects_dir + subject + '/' +
                                   subject + '-5120-bem-sol.fif', bem_sol)
        #(but is this to only create the ConductorModel?????????? Can't see how this is used in MNE.)

    # Computing the actual BEM solution
    for participant in participants
        3        # add patch statistics (used in depth wiegthing)
        #        mv $BEM_DIR / sample-oct-6-src.fif $BEM_DIR / sample-oct-6-orig-src.fif
        #        mne_add_patch_info --dist 7 --src $BEM_DIR / sample-oct-6-orig-src.fif --srcp $BEM_DIR / sample-oct-6-src.fif
        #        done
        #... BE SURE TO ADD CORTICAL PATCH STATISTICS and to copy exactly what was in the origional BEM script

    # To deleat in refactor
    # Remember not to use the medial wall (etc) Surface
    #        # creates surfaces necessary for BEM head models
    #       mne_watershed_bem --overwrite --subject ${subjects[m]}
    #        ln -s $SUBJECTS_DIR / ${subjects[m]} / bem / watershed / ${subjects[m]}'_inner_skull_surface' $SUBJECTS_DIR / ${subjects[m]} / bem / inner_skull.surf
    #        ln -s $SUBJECTS_DIR / ${subjects[m]} / bem / watershed / ${subjects[m]}'_outer_skull_surface' $SUBJECTS_DIR / ${subjects[m]} / bem / outer_skull.surf
    #        ln -s $SUBJECTS_DIR / ${subjects[m]} / bem / watershed / ${subjects[m]}'_outer_skin_surface'  $SUBJECTS_DIR / ${subjects[m]} / bem / outer_skin.surf
    #        ln -s $SUBJECTS_DIR / ${subjects[m]} / bem / watershed / ${subjects[m]}'_brain_surface'       $SUBJECTS_DIR / ${subjects[m]} / bem / brain_surface.surf
    ##        # create acurate head model for coregistration
    ##        mkheadsurf -s ${subjects[m]}
    ##        mne_surf2bem --surf $SUBJECTS_DIR / ${subjects[m]} / surf / lh.seghead --id 4 --check --fif $SUBJECTS_DIR / ${subjects[m]} / bem / ${subjects[m]}-head.fif
    #        # creates fiff-files for MNE describing MRI data
    #        mne_setup_mri --overwrite --subject ${subjects[m]}
    #        # create a source space from the cortical surface created in Freesurfer
    ##        mne_setup_source_space --ico 5 --overwrite --subject ${subjects[m]}
    #
    #    If your BEM meshes do not look correct when
    #    viewed in mne.viz.plot_alignment() or mne.viz.plot_bem(), consider potential solutions from the FAQ.

def create_forward_model_and_inverse_solution():

    # Compute forward solution
    for participant in participants
        http://martinos.org/mne/stable/auto_examples/forward/plot_make_forward.html#sphx-glr-auto-examples-forward-plot-make-forward-py

        ... BE SURE TO copy exactly what was in the origional FORWARD script

    # CHECK SENSITIVVITY MAPS
    for participant in participants
        # mne.sensitivity_map
        ... http://martinos.org/mne/stable/auto_examples/forward/plot_make_forward.html#sphx-glr-auto-examples-forward-plot-make-forward-py

    # Compute inverse operator
    for participant in participants
        http://martinos.org/mne/stable/auto_examples/inverse/plot_make_inverse_operator.html#sphx-glr-auto-examples-inverse-plot-make-inverse-operator-py

        ... BE SURE TO copy exactly what was in the origional create operator script

        #    # Apply maxwell filtering (and everything else, such as filtering) to the empty room
        #
        #    .maxwell_filter_prepare_emptyroom,
        #    .maxwell_filter

        # USe Empty room Max filtered for covarience!
        # USe em[ty room for MEG and dia for EEG etc] (is on MNE)

def create_hexel_current_files():

    snr = 1
    lambda2 = 1.0 / snr ** 2

    for p in participants:

        # First compute morph matices for participant
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
        inverse_operator = read_inverse_operator((data_path + '3-sensor-data/inverse-operators/' + p + '_ico-5-3L-loose02-diagnoise-nodepth-reg-inv-csd.fif'))

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
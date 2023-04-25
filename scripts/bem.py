# BEMS


https://mne.tools/stable/auto_tutorials/forward/10_background_freesurfer.html
mne.gui.coregistration().
https://mne.tools/stable/auto_tutorials/forward/20_source_alignment.html

mne.viz.plot_bem()

# PLot eeg an MEG locations for Sanity checks
# https://mne.tools/stable/auto_examples/visualization/eeg_on_scalp.html#ex-eeg-on-scalp
# https://mne.tools/stable/auto_examples/visualization/meg_sensors.html#ex-plot-meg-sensors
# % One-offs
# NKG_make_morph_maps;
mne.bem.make_watershed_bem
mne.bem.make_scalp_surfaces
mne.sensitivity_map
mne.viz.plot_alignment
# NKG_make_averaged_meshes; (use FS average?)
# NKG_make_labels.sh (needed to remove later)
# Remember not to use the medial wall (etc) Surface

def create_boundary_element_model(config:dict):
    '''Create the bondary element models'''

    list_of_participants = config['list_of_participants']

    input("Please make sure you have run (in the terminal, for all participanta)"
          "the xxxx (to xxxx), and the MRI_xxx (to convert the xxx to thexxx)"
          "commands). You should then have XYZ to xxx./n"
          "For more details : https://mne.tools/stable/auto_tutorials/forward/10_background_freesurfer.html)"


    # creat BEMS

    for participant in list_of_participants:

        mne.bem.make_watershed_bem()
        make_scalp_surfaces ???
        mne.bem.make_flash_bem().



        # creates surfaces necessary for BEM head models
        mne_watershed_bem --overwrite --subject ${subjects[m]}
        ln -s $SUBJECTS_DIR / ${subjects[m]} / bem / watershed / ${subjects[m]}'_inner_skull_surface' $SUBJECTS_DIR / ${subjects[m]} / bem / inner_skull.surf
        ln -s $SUBJECTS_DIR / ${subjects[m]} / bem / watershed / ${subjects[m]}'_outer_skull_surface' $SUBJECTS_DIR / ${subjects[m]} / bem / outer_skull.surf
        ln -s $SUBJECTS_DIR / ${subjects[m]} / bem / watershed / ${subjects[m]}'_outer_skin_surface'  $SUBJECTS_DIR / ${subjects[m]} / bem / outer_skin.surf
        ln -s $SUBJECTS_DIR / ${subjects[m]} / bem / watershed / ${subjects[m]}'_brain_surface'       $SUBJECTS_DIR / ${subjects[m]} / bem / brain_surface.surf
        # create acurate head model for coregistration
        mkheadsurf -s ${subjects[m]}
        mne_surf2bem --surf $SUBJECTS_DIR / ${subjects[m]} / surf / lh.seghead --id 4 --check --fif $SUBJECTS_DIR / ${subjects[m]} / bem / ${subjects[m]}-head.fif
        # creates fiff-files for MNE describing MRI data
        mne_setup_mri --overwrite --subject ${subjects[m]}
        # create a source space from the cortical surface created in Freesurfer
        mne_setup_source_space --ico 5 --overwrite --subject ${subjects[m]}

        # add patch statistics (used in depth wiegthing)
        mv $BEM_DIR / sample-oct-6-src.fif $BEM_DIR / sample-oct-6-orig-src.fif
        mne_add_patch_info --dist 7 --src $BEM_DIR / sample-oct-6-orig-src.fif --srcp $BEM_DIR / sample-oct-6-src.fif
        done

    # Check the BEMS look OK using

    mne.viz.plot_alignment() or mne.viz.plot_bem(),

    If your BEM meshes do not look correct when
    viewed in mne.viz.plot_alignment() or mne.viz.plot_bem(), consider potential solutions from the FAQ.


    # start alignment (replaces tkedit?)

    for participant in list_of_participants:
        mne.gui.coregistration()
        Creates a trans file

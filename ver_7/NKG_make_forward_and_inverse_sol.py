#!/usr/bin/python

import sys
sys.path.append('/imaging/local/software/mne_python/v0.9full')
sys.path.append('/imaging/local/software/python_packages/scikit-learn/v0.14.1')
sys.path.append('/imaging/local/software/python_packages/pysurfer/v0.4')
import os
import pdb
import pylab as pl
import mne
import numpy as np
from mne import read_evokeds
from mne.minimum_norm import apply_inverse, read_inverse_operator


participants = [
                    'meg15_0045',
                    'meg15_0051',
                    'meg15_0054',
                    'meg15_0055',
                    'meg15_0056',
                    'meg15_0058',
                    'meg15_0060',
                    'meg15_0065',
                    'meg15_0066',
                    'meg15_0070',
                    'meg15_0071',
                    'meg15_0072',
                    'meg15_0079',
                    'meg15_0081',
                    'meg15_0082', 
]
inputstream = 'visual'


1. start up
To compute a forward operator we need:
a -trans.fif file that contains the coregistration info - Should have done this in mne_anayse already.
a source space // to be created...
the BEM surfaces // to be created ...

To have run corregistaration

// import these...

2.
//Compute and visualize BEM surfaces

//http://martinos.org/mne/stable/auto_examples/forward/plot_bem_contour_mri.html#sphx-glr-auto-examples-forward-plot-bem-contour-mri-py	

from mne.viz import plot_bem
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'

plot_bem(subject='sample', subjects_dir=subjects_dir, orientation='axial')
plot_bem(subject='sample', subjects_dir=subjects_dir, orientation='sagittal')
plot_bem(subject='sample', subjects_dir=subjects_dir, orientation='coronal')

// we can plot them in three3d - but not really necasarry.

- http://martinos.org/mne/stable/auto_examples/forward/plot_read_bem_surfaces.html#sphx-glr-auto-examples-forward-plot-read-bem-surfaces-py

3.
//Visualization the coregistration


condition = 'Left Auditory'
evoked = read_evokeds(evoked_fname, condition=condition, baseline=(-0.2, 0.0))
plot_trans(evoked.info, trans_fname, subject='sample', dig=True,
           subjects_dir=subjects_dir)

4. create the source SPACE

Compute Source Space - 
The source space defines the position of the candidate source locations.
The following code compute such a source space with an OCT-6 resolution.
In [5]:
mne.set_log_level('WARNING')
subject = 'sample'
src = mne.setup_source_space(subject, spacing='oct6',
                             subjects_dir=subjects_dir,
                             add_dist=False, overwrite=True)
In [6]:
print(src)
<SourceSpaces: [<'surf', n_vertices=155407, n_used=4098, coordinate_frame=MRI (surface RAS)>, <'surf', n_vertices=156866, n_used=4098, coordinate_frame=MRI (surface RAS)>]>
src contains two parts, one for the left hemisphere (4098 locations) and one for the right hemisphere (4098 locations).
Let's write a few lines of mayavi to see what it contains
In [7]:
import numpy as np
from surfer import Brain

brain = Brain('sample', 'lh', 'inflated', subjects_dir=subjects_dir)
surf = brain._geo

vertidx = np.where(src[0]['inuse'])[0]

mlab.points3d(surf.x[vertidx], surf.y[vertidx],
              surf.z[vertidx], color=(1, 1, 0), scale_factor=1.5)

mlab.savefig('source_space_subsampling.jpg')
Image(filename='source_space_subsampling.jpg', width=500)

... BE SURE TO copy exactly what was in the origional SOURCE SPACE script (or improve)

5. MAKE BEM MODEL - is using this better than the default????

conductivity = (0.3,)  # for single layer
# conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(subject='sample', ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

bem.writebem - but is this to only create the ConductorModel?????????? Can't see how this is used in MNE.

... BE SURE TO ADD CORTICAL PATCH STATISTICS and to copy exactly what was in the origional BEM script

HOW TO DEAL WITH OVERLAPS?????/MISSIZES

6. Compute forward solution

http://martinos.org/mne/stable/auto_examples/forward/plot_make_forward.html#sphx-glr-auto-examples-forward-plot-make-forward-py

... BE SURE TO copy exactly what was in the origional FORWARD script

7. CHECK SENSITIVVITY MAPS

... http://martinos.org/mne/stable/auto_examples/forward/plot_make_forward.html#sphx-glr-auto-examples-forward-plot-make-forward-py

6. Compute inverse operator

http://martinos.org/mne/stable/auto_examples/inverse/plot_make_inverse_operator.html#sphx-glr-auto-examples-inverse-plot-make-inverse-operator-py

... BE SURE TO copy exactly what was in the origional create operator script


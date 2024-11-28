#!/bin/bash

###
# To run gridsearch locally, run the following command in command line:
#   ./submit_gridsearch.sh (might check whether need to add bash before this)
###



poetry run python -m invokers.run_gridsearch \
  --config dataset4.yaml \
  --input-stream auditory \
  --function-path 'predicted_function_contours/GMSloudness/stimulisig' \
  --function-name IL STL IL1 IL2 IL3 IL4 IL5 IL6 IL7 IL8 IL9  \
  --plot-top-channels \
  --overwrite

# current terminology change: function -> transform
# IL1-IL9: frequency band from low to high
# can open up "stimulisig.npz" to see the functions

# in output folder, the other 11 png files have more information. it's like a sanity (debug) check, but most of
# its information will not be written into the nkg file (nkg file only save the best matches/numbers)
# this is what 'plot-top-channels' does


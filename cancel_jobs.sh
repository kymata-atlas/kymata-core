#!/bin/bash

# Define the range of job IDs
start_id=3543205_8
end_id=3543205_33

# Loop through the range of job IDs and cancel each job
for ((i=${start_id#*_}; i<=${end_id#*_}; i++)); do
    job_id="${start_id%_*}_$i"
    scancel "$job_id"
done
#!/bin/bash


### PARAMS INI ###

# this should be the path for the output files (choose YOUR OWN dir!)
out_path="/Users/karinemiras/projects/evogym/tmp_out"

# DO NOT use underline ( _ ) in the study and experiments names
# delimiter of three vars below is coma. example:
#experiments="exp1,epx2"
# exps order is the same for all three vars
# exps names should not be fully contained in each other

study_name="foraging"
experiments="foraging"

# one set of conditions per experiment
env_conditions="none"

####

nruns=1

runs=""
for i in $(seq 1 $nruns);
do
  runs="${runs}${i},"
done
runs="${runs%,}"

watchruns=$runs

algorithm="foraging_customEA"

fitness_metric="reward"

plastic=0

num_generations="30"

population_size="50"

offspring_size="25"

# gens for box-plots, snapshots, videos (by default the last gen)
#generations="1,$num_generations"
generations="$num_generations"

# max gen to filter line-plots  (by default the last gen)
final_gen="$num_generations"

mutation_prob=0.9

crossover_prob=1

max_voxels=36

cube_face_size=6

evogym_num_workers=0

run_simulation=1

### PARAMS END ###

#!/bin/bash --login
## Note that the default queue seems to have a time limit of 1 day
#SBATCH --job-name="Cr8"
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm/output.txt
#SBATCH --error=slurm/error.txt
#SBATCH --partition=compute
##SBATCH --mail-type=ALL
##SBATCH --mail-user=elia.stocco01@universitadipavia.it
#SBATCH --exclusive

source ~/.bashrc
source ~/.elia

#elia libraries for quantum espresso
echo
RUN="true"
echo "running : $RUN"
if [[ $RUN == "true" ]]; then
   echo
   eval "$cmd"

fi


echo
echo "Starting script"

    OUT_DIR="./outdir"
RESULTS_DIR="./results"

for DIR in $OUT_DIR $RESULTS_DIR ; do
    if test ! -d $DIR ; then
        mkdir $DIR
    fi
done

PREFIX="Cr8"
CALC="scf"
INPUT_FILE="$PREFIX.$CALC.in"
OUTPUT_FILE="$PREFIX.$CALC.out"


# PW
PW_COMMAND="mpirun -np 32 ${BIN_DIR}/pw.x < $INPUT_FILE > $OUTPUT_FILE"


echo "output  directory : $OUT_DIR"
echo "results directory : $RESULTS_DIR"
echo "pseudo  directory : $PSEUDO_DIR"
echo "binary  directory : $BIN_DIR"
echo
echo "------------------------------------------------------"
echo
echo "      PW"
echo
echo "           prefix : $PREFIX"
echo "      input  file : $INPUT_FILE"
echo "      output file : $OUTPUT_FILE"
echo



occupations="fixed"

#startingpot="file"
#startingwfc="file"
#restart_mode="restart"

restart_mode="from_scratch"
startingpot="atomic"
startingwfc="atomic+random"

ratio="8"
ecutwfc="80"
ecutrho=$((ecutwfc*ratio))
electron_maxstep="300"
conv_thr="1.0D-5"
max_seconds=$((4*3600-400))


short_name="$PREFIX"
full_name="$short_name"
OUT_DIR_CYCLE="$OUT_DIR/$short_name"


echo
echo "---------------------------"
echo
echo "  startingwfc = $startingwfc"
echo "  startingpot = $startingpot"
echo "  occupations = $occupations"
echo "   projection = $projection"
echo "configuration = $configuration"
echo
echo "   short name = $short_name"
echo "    full name = $full_name"
echo "       outdir = $OUT_DIR_CYCLE"


startdate=$(date +%s)
echo

echo "writing input file"

#######################
cat > $INPUT_FILE << EOF
&CONTROL
  calculation ='$CALC'
  restart_mode='${restart_mode}'
  prefix      ='$PREFIX'
  pseudo_dir  ='$PSEUDO_DIR'
  outdir      ='$OUT_DIR_CYCLE'
  tprnfor = .true.
  verbosity = 'low'
  max_seconds =${max_seconds}
/
&SYSTEM
  ecutrho =   400
  ecutwfc =   50
  ibrav = 0
  nat = 2
  ntyp = 3
  occupations = '${occupations}'
/
&ELECTRONS
  mixing_mode = 'TF'
  conv_thr = ${conv_thr}
  mixing_beta = 0.5
  mixing_ndim=25
  electron_maxstep=${electron_maxstep}
  startingpot = '$startingpot'
  startingwfc = '$startingwfc'
/
ATOMIC_SPECIES
Li     24.305 Mg.pbesol-n-kjpaw_psl.0.3.0.UPF
Nb     15.9994 O.pbesol-n-kjpaw_psl.0.1.UPF
O
ATOMIC_POSITIONS crystal
Li 0.140367 0.140397 0.140433
Li 0.390428 0.390436 0.390379
Nb 0.499045 0.499072 0.499034
Nb 0.249071 0.249055 0.249024
O  0.052928 0.182854 0.358316
O  0.358314 0.052924 0.182871
O  0.182849 0.358293 0.052932
O  0.108312 0.432847 0.302929
O  0.432856 0.302927 0.108312
O  0.302928 0.108294 0.432870
Li 0.140367 0.140397 0.640433
Li 0.390428 0.390436 0.890379
Nb 0.499045 0.499072 0.999034
Nb 0.249071 0.249055 0.749023
O  0.052928 0.182854 0.858316
O  0.358314 0.052924 0.682871
O  0.182849 0.358293 0.552932
O  0.108312 0.432847 0.802929
O  0.432856 0.302927 0.608312
O  0.302928 0.108294 0.932870
Li 0.140367 0.640397 0.140433
Li 0.390428 0.890436 0.390379
Nb 0.499045 0.999072 0.499034
Nb 0.249071 0.749055 0.249024
O  0.052928 0.682854 0.358316
O  0.358315 0.552924 0.182871
O  0.182849 0.858293 0.052932
O  0.108312 0.932847 0.302929
O  0.432856 0.802927 0.108312
O  0.302928 0.608294 0.432870
Li 0.140367 0.640397 0.640433
Li 0.390428 0.890436 0.890379
Nb 0.499045 0.999072 0.999034
Nb 0.249071 0.749055 0.749023
O  0.052928 0.682854 0.858316
O  0.358315 0.552924 0.682871
O  0.182849 0.858293 0.552932
O  0.108312 0.932847 0.802929
O  0.432856 0.802927 0.608312
O  0.302928 0.608294 0.932870
Li 0.640367 0.140397 0.140433
Li 0.890428 0.390436 0.390379
Nb 0.999045 0.499072 0.499034
Nb 0.749071 0.249055 0.249024
O  0.552928 0.182854 0.358316
O  0.858314 0.052924 0.182871
O  0.682849 0.358293 0.052932
O  0.608312 0.432847 0.302929
O  0.932856 0.302927 0.108312
O  0.802928 0.108294 0.432870
Li 0.640367 0.140397 0.640433
Li 0.890427 0.390436 0.890379
Nb 0.999045 0.499072 0.999034
Nb 0.749071 0.249055 0.749023
O  0.552928 0.182854 0.858316
O  0.858314 0.052924 0.682871
O  0.682849 0.358293 0.552932
O  0.608312 0.432847 0.802929
O  0.932856 0.302927 0.608312
O  0.802928 0.108294 0.932870
Li 0.640367 0.640397 0.140433
Li 0.890428 0.890436 0.390379
Nb 0.999045 0.999072 0.499034
Nb 0.749071 0.749055 0.249024
O  0.552928 0.682854 0.358316
O  0.858315 0.552924 0.182871
O  0.682849 0.858293 0.052932
O  0.608312 0.932847 0.302929
O  0.932856 0.802927 0.108312
O  0.802928 0.608294 0.432870
Li 0.640367 0.640397 0.640433
Li 0.890428 0.890436 0.890379
Nb 0.999045 0.999072 0.999034
Nb 0.749071 0.749055 0.749023
O  0.552928 0.682854 0.858316
O  0.858315 0.552924 0.682871
O  0.682849 0.858293 0.552932
O  0.608312 0.932847 0.802929
O  0.932856 0.802927 0.608312
O  0.802928 0.608294 0.932870

K_POINTS automatic
  ${nk} ${nk} ${nk} 0 0 0
CELL_PARAMETERS bohr
  10.849570 0.000000 0.000000
  6.087115 8.981102 0.000000
  6.086463 3.227500 8.380107

EOF

########################
#PW

echo
echo "evaluating command: $RUN"
echo "command: $PW_COMMAND"
if [[ $RUN == "true" ]]; then
   eval "$PW_COMMAND"
fi

echo
echo "copying input file: $RUN"
if [[ $RUN == "true" ]]; then
   COPY_INPUT_FILE="cp $INPUT_FILE $RESULTS_DIR/$full_name.$CALC.in"
   echo "$COPY_INPUT_FILE"
   eval "$COPY_INPUT_FILE"
fi

echo
echo "copying output file: $RUN"
if [[ $RUN == "true" ]]; then
   COPY_OUTPUT_FILE="cp $OUTPUT_FILE $RESULTS_DIR/$full_name.$CALC.out"
   echo "$COPY_OUTPUT_FILE"
   eval "$COPY_OUTPUT_FILE"
fi


########################

finaldate=$(date +%s)
runtime=$((finaldate-startdate))
echo
echo "final time : $finaldate"
echo
echo "elapsed time: $runtime seconds"
echo

########################

echo
echo "---------------------------"
echo
echo "Finished"
echo

#!/bin/bash --login

source ~/.bashrc
source ~/.elia

if [ -z ${VAR_SOURCED+x} ]; then
  echo  "souring var.sh"
  source var.sh
fi

#elia libraries for quantum espresso
echo
RUN="true"
echo  "running : $RUN"
# if [[ $RUN == "true" ]]; then
#    echo
#    eval "$cmd"
#
# fi


echo
echo  "Starting script"

OUT_DIR="./outdir"
RESULTS_DIR="./results"

for DIR in ${OUT_DIR} ${RESULTS_DIR} ; do
    if test ! -d ${DIR} ; then
        mkdir ${DIR}
    fi
done

VAR_SOURCED="ciao"
PREFIX="LiNbO3"
CALC="hp"
INPUT_FILE="$PREFIX.$CALC.in"
OUTPUT_FILE="$PREFIX.$CALC.out"

QE_COMMAND="${QE_PATH}/hp.x < $INPUT_FILE > $OUTPUT_FILE"


echo  "output  directory : $OUT_DIR"
echo  "results directory : $RESULTS_DIR"
echo  "pseudo  directory : $PSEUDO_DIR"
echo  "binary  directory : $BIN_DIR"
echo
echo  "------------------------------------------------------"
echo
echo  "      PW"
echo
echo  "           prefix : $PREFIX"
echo  "      input  file : $INPUT_FILE"
echo  "      output file : $OUTPUT_FILE"
echo


#startingpot="file"
#startingwfc="file"
#restart_mode="restart"

restart_mode="from_scratch"
startingpot="atomic"
startingwfc="atomic+random"

#occupations="fixed"
occupations="smearing"

ratio="8"
ecutwfc="80"
ecutrho=$((ecutwfc*ratio))
electron_maxstep="100"
conv_thr="1.0D-5"
max_seconds=$((4*3600-400))
nk="2"


short_name="$PREFIX"
full_name="$short_name"
OUT_DIR_CYCLE="$OUT_DIR/$short_name"


echo
echo  "---------------------------"
echo
echo  "  startingwfc = $startingwfc"
echo  "  startingpot = $startingpot"
echo  "  occupations = $occupations"
echo  "   projection = $projection"
echo  "configuration = $configuration"
echo
echo  "   short name = $short_name"
echo  "    full name = $full_name"
echo  "       outdir = $OUT_DIR_CYCLE"

echo

echo  "writing input file"

#######################
cat > ${INPUT_FILE} << EOF
&inputhp
   prefix = '$PREFIX',
   outdir='$OUT_DIR_CYCLE'
   find_atpert=2
   skip_equivalence_q=.false.
   niter_max = 200
   conv_thr_chi = 1.D-5
   equiv_type(1)=2
   nq1=2
   nq2=2
   nq3=2
   !dist_thr=1.0
   !perturb_only_atom(1)=.true.
/

EOF

########################

echo
echo "evaluating command: $RUN"
echo "command: $QE_COMMAND"
if [[ $RUN == "true" ]]; then
   eval "$QE_COMMAND"
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

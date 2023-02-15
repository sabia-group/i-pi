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

for DIR in ${OUT_DIR} ; do
    if test ! -d ${DIR} ; then
        mkdir ${DIR}
    fi
done


echo  "output  directory : $OUT_DIR"
echo  "results directory : $RESULTS_DIR"
echo  "pseudos directory : $PSEUDO_DIR"
#echo  "binary  directory : $BIN_DIR"
echo
echo  "------------------------------------------------------"
echo
echo  "      PW"
echo
echo  "           prefix : $PREFIX"
echo  "      input  file : $INPUT_FILE"
echo  "      output file : $OUTPUT_FILE"
echo

#restart_mode="restart"
startingpot="file"
startingwfc="file"

restart_mode="from_scratch"
#startingpot="atomic"
#startingwfc="atomic+random"

#occupations="fixed"
occupations="smearing"

ratio="4"
ecutwfc="80"
ecutrho=$((ecutwfc*ratio))
electron_maxstep="100"
conv_thr="1.0D-5"
max_seconds=$((4*3600-400))
nk="3"
U_Ni="1.6"


short_name="$PREFIX.nk=${nk}"
full_name="$short_name.ecutwfc=${ecutwfc}.ratio=${ratio}.occ=${occupations}"
OUT_DIR_CYCLE="$OUT_DIR/$short_name"


echo
echo  "---------------------------"
echo
echo  "  startingwfc = $startingwfc"
echo  "  startingpot = $startingpot"
echo  "  occupations = $occupations"
#cho  "   projection = $projection"
#echo  "configuration = $configuration"
echo
echo  "   short name = $short_name"
echo  "    full name = $full_name"
echo  "       outdir = $OUT_DIR_CYCLE"

echo

echo  "writing input file"

#######################
cat > ${INPUT_FILE} << EOF
&CONTROL
  calculation = '$CALC'
  restart_mode= '${restart_mode}'
  prefix      = '${PREFIX}'
  pseudo_dir  = '${PSEUDO_DIR}'
  outdir      = '${OUT_DIR_CYCLE}'
  tprnfor     = .true.
  tstress     = .true.
  verbosity   = 'high'
  max_seconds = ${max_seconds}
/
&SYSTEM
  ecutwfc = ${ecutwfc}
  ecutrho = ${ecutrho}
  ibrav = 0
  nat = 10
  ntyp = 3
  !nspin=2
  occupations = '${occupations}'
  smearing='gauss'
  degauss=0.005
  !starting_magnetization(1)=0.0
  !starting_magnetization(2)=0.0
  !tot_magnetization=0.0
  !vdw_corr = 'grimme-d3'
/
&ELECTRONS
  mixing_mode = 'local-TF'
  conv_thr = ${conv_thr}
  mixing_beta = 0.9
  mixing_ndim=25
  electron_maxstep=${electron_maxstep}
  startingpot = '$startingpot'
  startingwfc = '$startingwfc'
/
ATOMIC_SPECIES
  Nb  92.90638 ${Nb_PSEUDO}
  Li  6.941    ${Li_PSEUDO}
  O   15.9994  ${O_PSEUDO}

ATOMIC_POSITIONS angstrom
Nb    4.181958    6.092849   11.489685
Nb    2.086843    3.040516    5.734021
Li    1.176843    1.714167    3.232280
Li    3.271417    4.766494    8.988627
 O    3.002726    2.798695    3.868177
 O    1.532483    1.065538    5.322757
 O    0.443576    3.388700    4.486975
 O    2.538581    4.865153    5.653706
 O    0.907666    3.070195    7.199496
 O    3.627496    2.369687    6.580484

K_POINTS automatic
  ${nk} ${nk} ${nk} 0 0 0

CELL_PARAMETERS angstrom
  4.190053 1.613750 3.043232
  0.000000 4.490551 3.043558
  0.000000 0.000000 5.424785

HUBBARD ortho-atomic
  U Nb-4d ${U_Ni}

EOF

########################

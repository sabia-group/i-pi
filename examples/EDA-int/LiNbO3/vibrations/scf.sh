#!/bin/bash --login

source ~/.bashrc
source ~/.elia

CALC="scf"

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
RES_DIR="./results"

for DIR in ${OUT_DIR} ${RES_DIR} ; do
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

occupations="fixed"
#occupations="smearing"

ratio="4"
ecutwfc="80"
ecutrho=$((ecutwfc*ratio))
electron_maxstep="100"
conv_thr="1.0D-5"
max_seconds=$((4*3600-400))
nk="3"
U_Ni="1.6"
gdir="1"
nppstr="3"


short_name="$PREFIX.nk=${nk}"
full_name="$short_name.occ=${occupations}.vibrations"
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
  !tprnfor     = .true.
  !tstress     = .true.
  verbosity   = 'high'
  !max_seconds = ${max_seconds}
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
  Nb 4.2298305249 6.1625038592 11.6257081907 
  Nb 2.1093164478 3.0730881001 5.7974466747
  Li 1.1891975655 1.7325908180 3.2685524330
  Li 3.3097489344 4.8220441236 9.0968872665
  O 3.0462237209 2.8225284608 3.9268775609
  O 1.5386925881 1.0847905761 5.3964858534
  O 0.4590824818 3.4413320635 4.5400960889
  O 2.5796095508 4.9152298070 5.7331053280
  O 0.9257086964 3.1211034601 7.2858904276
  O 3.6591977211 2.4018176340 6.6726553421

K_POINTS automatic
  ${nk} ${nk} ${nk} 0 0 0

CELL_PARAMETERS angstrom
  4.24102999660546 1.63392996111767 3.08247152070467
  0.00000000000000 4.54490544781247 3.08244612716667
  0.00000000000000 0.00000000000000 5.49157234881015

HUBBARD ortho-atomic
  U Nb-4d ${U_Ni}

EOF

########################

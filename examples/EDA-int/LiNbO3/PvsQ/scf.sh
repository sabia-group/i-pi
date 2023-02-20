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
#startingpot="file"
#startingwfc="file"

restart_mode="from_scratch"
startingpot="atomic"
startingwfc="atomic+random"

occupations="fixed"
#occupations="smearing"

ratio="4"
ecutwfc="80"
ecutrho=$((ecutwfc*ratio))
electron_maxstep="100"
conv_thr="1.0D-4"
max_seconds=$((4*3600-400))
nk="2"
U_Ni="1.6"
gdir="${1}"
nppstr="3"


short_name="$PREFIX.nk=${nk}"
full_name="$short_name.ecutwfc=${ecutwfc}.ratio=${ratio}.occ=${occupations}.gdir=${gdir}.nppstr=${nppstr}.shift=${2}"
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
  !tstress     = .false.
  verbosity   = 'high'
  max_seconds = ${max_seconds}
  lberry      = .true.
  gdir        = ${gdir}
  nppstr      = ${nppstr}
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

K_POINTS automatic
  ${nk} ${nk} ${nk} 0 0 0

CELL_PARAMETERS angstrom
  4.24102999660546 1.63392996111767 3.08247152070467
  0.00000000000000 4.54490544781247 3.08244612716667
  0.00000000000000 0.00000000000000 5.49157234881015

HUBBARD ortho-atomic
  U Nb-4d ${U_Ni}

ATOMIC_POSITIONS angstrom
EOF

(tail -n +3 shifted.xyz) >> ${INPUT_FILE}

########################

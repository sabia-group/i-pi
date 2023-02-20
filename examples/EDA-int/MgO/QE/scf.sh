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
conv_thr="1.0D-5"
max_seconds=$((4*3600-400))
nk="2"
#U_Ni="1.6"


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
  tstress     = .false.
  verbosity   = 'high'
  max_seconds = ${max_seconds}
  lberry        = .true.
  gdir          = 1
  nppstr        = 2
/
&SYSTEM
  ecutwfc = ${ecutwfc}
  ecutrho = ${ecutrho}
  ibrav = 0
  nat = 2
  ntyp = 2
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
  Mg  24.305   ${Mg_PSEUDO}
  O   15.9994  ${O_PSEUDO}

ATOMIC_POSITIONS angstrom
  Mg           0.000000 0.000000 0.000000
  O            1.219997 1.725336 2.988370

K_POINTS automatic
  ${nk} ${nk} ${nk} 0 0 0

CELL_PARAMETERS angstrom
  2.4399938875       0.8626681120       1.4941850000
  0.0000000000       2.5880043359       1.4941850000
  0.0000000000       0.0000000000       2.9883700000

EOF

########################

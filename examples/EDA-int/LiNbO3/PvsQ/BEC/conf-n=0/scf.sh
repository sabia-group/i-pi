#!/bin/bash --login

source ~/.bashrc
source ~/.elia

#CALC="scf"
CALC="${2}"

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
U_Ni="1.6"
gdir="${3}"
nppstr="2"


short_name="$PREFIX.nk=${nk}.displaced"
echo "CALC = ${CALC}"
if [[ ${CALC} == "scf" ]]; then
  full_name="$short_name.occ=${occupations}.n=${1}"
  lberry=".false."
  yn="!"
else
  full_name="$short_name.occ=${occupations}.n=${1}.gdir=${gdir}"
  lberry=".true."
  yn=""
fi

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
  calculation = '${CALC}'
  restart_mode= '${restart_mode}'
  prefix      = '${PREFIX}'
  pseudo_dir  = '${PSEUDO_DIR}'
  outdir      = '${OUT_DIR_CYCLE}'
  !tprnfor     = .true.
  !tstress     = .true.
  verbosity   = 'high'
  !lelfield    = .true.
  !max_seconds = ${max_seconds}
  ${yn}lberry = ${lberry}
  ${yn}gdir = ${gdir}
  ${yn}nppstr = ${nppstr}

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

K_POINTS automatic
  ${nk} ${nk} ${nk} 0 0 0

CELL_PARAMETERS angstrom
 4.2442209319436959 1.6346114044445832 3.0825805539430258
 0.0000000000000000 4.5481196540733295 3.0825782703885887
 0.0000000000000000 0.0000000000000000 5.4943305901844859

HUBBARD ortho-atomic
  U Nb-4d ${U_Ni}

ATOMIC_POSITIONS angstrom
EOF

(tail -n +3 conf/configurations.n=${1}.xyz) >> ${INPUT_FILE}

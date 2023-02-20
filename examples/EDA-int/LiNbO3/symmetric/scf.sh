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
Nb 0.0000000000000000 0.0000000000000000 0.0000000000000000
Nb 2.1221095475093028 3.0913641912968206 5.8297421841084560
Li 0.9269379812055256 1.3503071218556133 2.5464318072137013
Li 3.0490472225606471 4.4416708671650555 8.3761731502722938
O  3.3084140913015978 3.0434134293594788 4.3767229121143147
O  0.5984328989453233 3.7757846260319821 4.9748369898250369
O  1.6510004045942073 1.2771466584192508 5.9166516041872290
O  2.7205454907139694 5.0910623021432997 6.2520634158725779
O  3.7731119747571396 2.7244134817558763 7.1238879101699180
O  1.1863007773106955 3.3722326019081628 7.7220021661886218

K_POINTS automatic
  ${nk} ${nk} ${nk} 0 0 0

CELL_PARAMETERS angstrom
 4.2442209319436959 1.6346114044445832 3.0825805539430258
 0.0000000000000000 4.5481196540733295 3.0825782703885887
 0.0000000000000000 0.0000000000000000 5.4943305901844859

HUBBARD ortho-atomic
  U Nb-4d ${U_Ni}

EOF

########################

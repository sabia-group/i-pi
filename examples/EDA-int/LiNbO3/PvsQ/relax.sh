#!/bin/bash --login

source ~/.bashrc
source ~/.elia

CALC="relax"

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
U_Ni="1.6"
gdir="1"
nppstr="3"

e_thr="1.0D-4"
f_thr="1.0D-3"


short_name="$PREFIX.nk=${nk}"
full_name="$short_name.occ=${occupations}.shifted"
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
  !max_seconds = ${max_seconds}
  etot_conv_thr = ${e_thr}
  forc_conv_thr = ${f_thr}
  /
&SYSTEM
  ecutwfc = ${ecutwfc}
  ecutrho = ${ecutrho}
  ibrav = 5
  celldm(1) = 10.38277894
  celldm(4) = 0.5610464837357743
  nat = 3
  ntyp = 3
  occupations = '${occupations}'
  smearing='gauss'
  degauss=0.005
  space_group = 161
/
&ELECTRONS
  mixing_mode = 'local-TF'
  conv_thr = ${conv_thr}
  !mixing_beta = 0.9
  !mixing_ndim=25
  electron_maxstep=${electron_maxstep}
  startingpot = '$startingpot'
  startingwfc = '$startingwfc'
/
&IONS
  upscale = 10
/
ATOMIC_SPECIES
  Nb  92.90638 ${Nb_PSEUDO}
  Li  6.941    ${Li_PSEUDO}
  O   15.9994  ${O_PSEUDO}

ATOMIC_POSITIONS crystal_sg
  Nb  0.0000000000000000000 0.0000000000000000000 0.0000000000000000000
  Li  0.2824001343871483647 0.2824000447459397976 0.2823998146073517499
  O   0.7895101996407776124 0.3989998985407385201 0.1509999082858564012

K_POINTS automatic
  ${nk} ${nk} ${nk} 0 0 0

HUBBARD ortho-atomic
  U Nb-4d ${U_Ni}

EOF

########################

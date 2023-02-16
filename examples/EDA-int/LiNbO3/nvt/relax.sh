#!/bin/bash --login

source ~/.bashrc
source ~/.elia

CALC="vc-relax"

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
conv_thr="1.0D-6"
max_seconds=$((4*3600-400))
nk="4"
U_Ni="1.6"

e_thr="1.0D-4"
f_thr="1.0D-3"


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
  !tprnfor     = .true.
  !tstress     = .false.
  verbosity   = 'high'
  max_seconds = ${max_seconds}
  etot_conv_thr = ${e_thr}
  forc_conv_thr = ${f_thr}
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
&IONS
  upscale = 10
/
&CELL
  cell_dofree = 'all'
/
ATOMIC_SPECIES
  Nb  92.90638 ${Nb_PSEUDO}
  Li  6.941    ${Li_PSEUDO}
  O   15.9994  ${O_PSEUDO}

ATOMIC_POSITIONS angstrom
Nb            4.2327299624        6.1651612807       11.6263575091
Nb            2.1107942565        3.0744650799        5.7978651387
Li            1.1903288875        1.7338307738        3.2696226140
Li            3.3123206203        4.8245653075        9.0982136375
O             3.0467711983        2.8229072845        3.9266076868
O             1.5396380143        1.0861394780        5.3962733177
O             0.4605211572        3.4419666711        4.5398983179
O             2.5814664707        4.9164327494        5.7324570392
O             0.9268120544        3.1229192912        7.2860004855
O             3.6605496637        2.4024733733        6.6727187175

K_POINTS automatic
  ${nk} ${nk} ${nk} 0 0 0

CELL_PARAMETERS angstrom
  4.241867807   1.634421791   3.082210455
  0.000991274   4.545783321   3.082409196
  0.000990844   0.001175729   5.492295302

HUBBARD ortho-atomic
  U Nb-4d ${U_Ni}

EOF

########################

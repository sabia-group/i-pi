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
nk="2"
U_Ni="1.6"
gdir="3"
nppstr="2"


short_name="$PREFIX.nk=${nk}.not-inverted"
full_name="$short_name.occ=${occupations}"
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
  !lelfield    = .true.
  !max_seconds = ${max_seconds}
  lberry = .true.
  gdir = ${gdir}
  nppstr = ${nppstr}

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
Nb    0.000000    0.000000    0.000000
Nb    2.122126    3.092350    5.824322
Li    0.926838    1.348442    2.530476
Li    3.048136    4.440824    8.365876
 O    3.320239    3.057040    4.360004
 O    0.577786    3.783015    4.975629
 O    1.657683    1.258873    5.922096
 O    2.701127    5.091096    6.256610
 O    3.781582    2.740497    7.107091
 O    1.199735    3.355142    7.722408

K_POINTS automatic
  ${nk} ${nk} ${nk} 0 0 0

CELL_PARAMETERS angstrom
 4.2442209319436959 1.6346114044445832 3.0825805539430258
 0.0000000000000000 4.5481196540733295 3.0825782703885887
 0.0000000000000000 0.0000000000000000 5.4943305901844859

HUBBARD ortho-atomic
  U Nb-4d ${U_Ni}

EOF

# symmetric
# Nb 0.0000000000000000 0.0000000000000000 0.0000000000000000
# Nb 2.1221095475093028 3.0913641912968206 5.8297421841084560
# Li 0.9269379812055256 1.3503071218556133 2.5464318072137013
# Li 3.0490472225606471 4.4416708671650555 8.3761731502722938
# O  3.3084140913015978 3.0434134293594788 4.3767229121143147
# O  0.5984328989453233 3.7757846260319821 4.9748369898250369
# O  1.6510004045942073 1.2771466584192508 5.9166516041872290
# O  2.7205454907139694 5.0910623021432997 6.2520634158725779
# O  3.7731119747571396 2.7244134817558763 7.1238879101699180
# O  1.1863007773106955 3.3722326019081628 7.7220021661886218

# up
# Nb           -0.0003161056        0.0002073887        0.0012275443
# Nb            2.1221934369        3.0921379312        5.8320321112
# Li            0.9256313581        1.3487047573        2.5410564873
# Li            3.0489707949        4.4411354571        8.3832313027
# O             3.3237317365        3.0560932588        4.3678681678
# O             0.5779953818        3.7829725478        4.9810835200
# O             1.6578768314        1.2575555730        5.9280634735
# O             2.6993248086        5.0906688017        6.2616451136
# O             3.7801168050        2.7424332400        7.1141581597
# O             1.2003753413        3.3554863244        7.7281462599

# dw
# Nb            0.0006461870        0.0001172059       -0.0060025640
# Nb            2.1221263044        3.0923502506        5.8243224036
# Li            0.9268380448        1.3484417653        2.5304762947
# Li            3.0481362182        4.4408235754        8.3658765019
# O             3.3202393749        3.0570401340        4.3600042350
# O             0.5777857977        3.7830146716        4.9756295204
# O             1.6576829374        1.2588725651        5.9220960503
# O             2.7011274916        5.0910964747        6.2566103836
# O             3.7815825376        2.7404967189        7.1070911095
# O             1.1997354953        3.3551419184        7.7224082049

########################

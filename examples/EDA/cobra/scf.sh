#!/bin/bash --login

source ~/.bashrc
source ~/.elia

CALC="scf"

if [ -z ${VAR_SOURCED+x} ]; then
  echo  "souring var.sh"
  source var.sh
fi

OUT_DIR="./outdir"
RES_DIR="./results"

for DIR in ${OUT_DIR} ${RES_DIR} ; do
    if test ! -d ${DIR} ; then
        mkdir ${DIR}
    fi
done

############################################################################
restart_mode="from_scratch"
startingpot="atomic"
startingwfc="atomic+random"

#restart_mode="restart"
startingpot="file"
startingwfc="file"

occupations="fixed"
#occupations="smearing"

ratio="8"
ecutwfc="100"
ecutrho=$((ecutwfc*ratio))
electron_maxstep="300"
conv_thr="1.0D-10"
max_seconds="$((2*3600-100))"
nk="2"

e_thr="1.0D-4"
f_thr="1.0D-3"

# c_bands eigenvalue not converged
# - increase k points
# - increase cutoff
# - change mixing_mode
# - lower mixing_beta (or higher?)
# - select 'cg' style diagonalization
# - modify 'mixing_ndim'

############################################################################

short_name="$PREFIX.nk=${nk}"
full_name="$short_name"
OUT_DIR_CYCLE="$OUT_DIR/$short_name"
#yn="!"

############################################################################
cat > ${INPUT_FILE} << EOF
&CONTROL
  calculation = '$CALC'
  restart_mode= '${restart_mode}'
  prefix      = '${PREFIX}'
  pseudo_dir  = '${PSEUDO_DIR}'
  outdir      = '${OUT_DIR_CYCLE}'
  tprnfor     = .true.
  !tstress     = .true.
  verbosity   = 'high'
  lelfield    = .true.
  !max_seconds = ${max_seconds}
  etot_conv_thr = ${e_thr}
  forc_conv_thr = ${f_thr}
/
&SYSTEM
  ecutwfc = ${ecutwfc}
  ecutrho = ${ecutrho}
  ibrav = 0
  nat   = 30
  ntyp  = 3
  occupations = 'fixed'
/
&ELECTRONS
  !mixing_mode = 'local-TF'
  !mixing_beta = 0.9
  mixing_ndim=25
  diagonalization = 'davidson' !ppcg, paro
  diago_david_ndim = 4
  conv_thr = ${conv_thr}
  ! diago_thr_init = 1.0D-4
  electron_maxstep=${electron_maxstep}
  startingpot = '${startingpot}'
  startingwfc = '${startingwfc}'
/
&IONS
  upscale = 10
  pot_extrapolation = 'second_order'
  wfc_extrapolation = 'second_order'
/
&CELL
  cell_dofree = 'all'

ATOMIC_SPECIES
  Nb  92.90638 ${Nb_PSEUDO}
  Li  6.941    ${Li_PSEUDO}
  O   15.9994  ${O_PSEUDO}

K_POINTS automatic
  ${nk} ${nk} ${nk} 0 0 0

CELL_PARAMETERS angstrom
  5.14495129661293 0.00000000000000 0.00000000000000
 -2.57248649065879 4.45568595877574 0.00000000000000
  0.00004096894127 0.00010785692501 13.85773788692639

ATOMIC_POSITIONS angstrom
Nb -0.0000013800 -0.0000285649 0.0117427132
Nb -0.0000054496 0.0000343398 6.9406116362
Nb 2.5724822014 1.4852383429 4.6309813811
Nb 2.5724842729 1.4852962150 11.5598499720
Nb 0.0000178983 2.9704964641 9.2502284589
Nb -0.0000330667 2.9704578235 2.3213589864
Li 0.0000260152 -0.0000051233 3.0243193969
Li -0.0000060507 0.0000623290 9.9531924549
Li 2.5725052251 1.4852519901 7.6435125392
Li 2.5724183417 1.4852232138 0.7146407492
Li 0.0000550074 2.9705171573 12.2627943781
Li -0.0000314505 2.9704919922 5.3339213264
O -0.6519245770 1.4329834293 1.4261731298
O 1.6574591900 3.1746893103 1.4262058749
O -1.0056149070 4.3038096591 1.4262376303
O 1.0056133247 4.3038465831 8.3550999964
O 0.6519292389 1.4330110074 8.3550644150
O -1.6574508297 3.1747140829 8.3550605432
O 1.9205751513 2.9182362954 6.0454266185
O 1.6574900040 0.2042685835 6.0454593737
O 4.1393700749 1.3333761399 6.0454722029
O 1.0056461852 1.3334289048 12.9743560074
O 3.2244135586 2.9182870808 12.9743043954
O 3.4875230615 0.2043041895 12.9743057068
O -0.6518906229 4.4034984434 10.6646711810
O -0.9149679498 1.6895243651 10.6647092863
O 1.5669062138 2.8186427386 10.6647211840
O -1.5668685448 2.8185946803 3.7358652384
O 0.6519080422 4.4034438238 3.7358143801
O 0.9150120376 1.6894637270 3.7358128587 

EOF

# CELL_PARAMETERS alat
#    0.468278700  -0.270360830   0.841201570
#    0.000000000   0.540721660   0.841201570
#   -0.468278700  -0.270360830   0.841201570

# Nb            0.0203476343        0.0203476343        0.0203476343
# Nb            0.5203476343        0.5203476343        0.5203476343
# Li            0.2375953375        0.2375951074        0.2375954272
# Li            0.7375954272        0.7375951074        0.7375953375
# O             0.7994916388        0.4096421916        0.1548332490
# O             0.1548332490        0.7994916388        0.4096421916
# O             0.4096421916        0.1548332490        0.7994916388
# O             0.6548332490        0.9096421916        0.2994916388
# O             0.9096421916        0.2994916388        0.6548332490
# O             0.2994916388        0.6548332490        0.9096421916

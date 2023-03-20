#!/bin/bash -l
# Standard output and error:
#SBATCH -o slurm/output.txt
#SBATCH -e slurm/error.txt
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J nve
## Queue (Partition):
#SBATCH --partition=general
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#
#SBATCH --mail-type=END
#SBATCH --mail-user=stocco@fhi-berlin.mpg.de
#
# Wall clock limit:
#SBATCH --time=08:00:00

# Settings from Archer2
# Choices here: short, standard, long,
#with max walltime of 20 min, 24 hours, 48 hours, respectively.
##SBATCH --qos=standard  #elia standard queue, comment if you want short
##SBATCH --qos=short  #elia short queue max 20 mins, if you use it, uncomment and uncomment the line below #SBATCH --reservation=shortqos
##SBATCH --qos=long   #elia don't change

#The shortqos reservation cannot be used without the short QoS. Please use the short QoS.
##SBATCH --reservation=shortqos   # uncomment if you want short queue


# TEMPLATE for running i-pi.
# 12 October 2020

source ~/.bashrc
source ~/.elia

relax='false'
run_ipi='false'
run_ipi_somewhereelse='true'
write_qe='true'
run_qe='true'
run_aims='false'
#sleep_sec="10"

if [[ ${run_ipi} == 'false' ]]; then
	sleep_sec="0"
fi

aims_radix='aims'
ipi_radix='i-pi'

if test -f RESTART; then
    ipi_input='RESTART'
else
    ipi_input='input.xml'
fi
echo "using '${ipi_input}' as input file for i-PI"

RESULTS_DIR="./results"

#PARA_PREFIX="srun -n 64"

#HOST=$(hostname)
#NTIME=21300

###################################################################
################## Setup input files ##############################
###################################################################


######### CHECK RESTART FILE ###############
#if [ ! -f "RESTART"   ]
# then
#  cp hessian.xml RESTART
#fi
######### CHECK RESTART FILE ###############

#echo  {"init",$( date -u)} >>LIST
#grep '<step>'  RESTART >> LIST

# Substitute the necessary SLURM variables in the input files
#sed -e "s:<address>.*:<address>$HOST</address>:" RESTART > RESTART.tmp1
#sed -e "s:<total_time>.*:<total_time>$NTIME</total_time>:" RESTART.tmp1 > RESTART.tmp2
#sed -e "s:<initialize nbeads='*'>:<initialize nbeads='1'>:" RESTART.tmp2 > ${ipi_input}
#sed -e "s/ipihost/$HOST/g" control.in.tmp > control.in

#rm RESTART.tmp1 RESTART.tmp2

###################################################################
################### Run the programs ##############################
###################################################################

# i-PI
if [[ ${run_ipi} == 'true' ]] ; then
	IPI_COMMAND="python -u ${IPI_PATH}/i-pi ${ipi_input} &> ${ipi_radix}.out &"
	echo "command: ${IPI_COMMAND}"
	eval "${IPI_COMMAND}"
fi

sleep_cmd="sleep ${sleep_sec}"
echo "command: ${sleep_cmd}"
eval "${sleep_cmd}"

# FHI-aims
#srun --exclusive -N 1 -n 96 --cpu_bind=rank --hint=nomultithread ${AIMSPATH}/${EXE} > ${aims_radix}.out
#srun --exclusive -N 2 -n 192 --cpu_bind=rank --hint=nomultithread ${AIMSPATH}/${EXE} > ${aims_radix}.out
if [[ ${run_aims} == 'true' ]] ; then
	AIMS_COMMAND="${PARA_PREFIX} ${AIMS_PATH}/${EXE} > ${aims_radix}.out"
	echo "command: ${AIMS_COMMAND}"
	eval "${AIMS_COMMAND}"
fi

# Quantum ESPRESSO
if [[ ${write_qe} == 'true' ]] ; then
	if [[ ${relax} == 'true' ]] ; then
		echo "sourcing var.sh from relax.sh"
		source relax.sh
	else
		echo "sourcing var.sh from scf.sh"
		source scf.sh
	fi
	if [ ! -z ${VAR_SOURCED+x} ]; then
		echo "sourcing var.sh from raven.sh"
	  source var.sh
	fi

	if [[ $run_qe == "true" ]]; then
		#QE_COMMAND="mpirun -np 32 ${QE_PATH}/pw.x < $INPUT_FILE > $OUTPUT_FILE"
		if [ ${run_ipi} == 'true' ] || [ ${run_ipi_somewhereelse} == 'true' ] ; then
			PARA_IPI="--ipi nve:UNIX"
		else
			PARA_IPI=""
		fi
		QE_COMMAND="${PARA_PREFIX} ${QE_PATH}/pw.x < $INPUT_FILE > $OUTPUT_FILE ${PARA_IPI}"
		echo "command: ${QE_COMMAND}"
		eval "${QE_COMMAND}"

		COPY_OUTPUT_FILE="cp $OUTPUT_FILE $RESULTS_DIR/$full_name.$CALC.out"
		echo "$COPY_OUTPUT_FILE"
		eval "$COPY_OUTPUT_FILE"

	fi

	echo
	COPY_INPUT_FILE="cp $INPUT_FILE $RESULTS_DIR/$full_name.$CALC.in"
	echo "$COPY_INPUT_FILE"
	eval "$COPY_INPUT_FILE"

fi

echo '1' >>count
l=`cat count|wc -l`

zip ${RESULTS_DIR}/results.${l}.t=0.5f.2K.zip ${ipi_radix}.* ${PREFIX}.* RESTART ${ipi_input} input.xml

#source ~/sleep4save.sh

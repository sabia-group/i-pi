# $LiNbO_3$

You should use the `EDA-int` branch of `i-PI`, and the EDA branch of `Quantum Espresso` (https://gitlab.com/eliastocco/q-e/-/tree/EDA).

If you want to use the `develop` branch of Quantum Espresso  you should also:

- specify `cpol=False` (*compute polarization*) in the `eda-nve.xml` input file

- `lelfield = .false.` in the Quantum Espresso input file (or just comment it out)

- remove all the *polarization-related* variable in the output properties of `eda-nve.xml`, such as `totalpol`, `EDAenergy` or `Eenthalpy`

The `BEC.txt` files has been produced by `ph.x` (it is necessary only for `eda-nve` calculations).

Just to let you know: each line contains the <u>transpose</u> of the BEC tensor of each ion (because `ph.x` prints these tensors in this way...).

You can type `./scf.sh` to print the `Quantum Espresso` input file.

You can use `cobra.sh` to run simulation in `cobra` (but you need to modify the file to make in work in your folder!).

Please specify in `cobra.sh` (or somewhere else) the path for the pseudopotentials (like `PSEUDO_DIR="./pseudos"`). Some pseudopotentials are indeed provided in the `pseudo` folder.

The script `cobra.sh` sources two other scripts, here they are (I do not put them in the folder to avoid confusion):

`.elia`

```bash

#!/bin/bash -l

# Intel OneAPI
source /home/elia/intel/oneapi/setvars.sh >/dev/null

# i-PI
# source /home/elia/i-pi-master/env.sh
#source ~/Google-Drive/google-personal/i-pi/env.sh

# quantum Espresso
#source /home/elia/q-e-master/environment_variables

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=false
ulimit -s unlimited

# https://vibes-developers.gitlab.io/vibes/Installation/
eval "$(_VIBES_COMPLETE=source vibes)"

# Quantum ESPRESSO

# ATTENTION: you should modify this
PSEUDO_DIR="/home/elia/Google-Drive/google-personal/pseudos"
QE_PATH="/home/elia/Google-Drive/google-personal/q-e/bin"

Li_PSEUDO="Li.pbesol-sl-rrkjus_psl.1.0.0.UPF"
Nb_PSEUDO="Nb.pbesol-spn-rrkjus_psl.1.0.0.UPF"
 O_PSEUDO="O.pbesol-n-rrkjus_psl.1.0.0.UPF"
Mg_PSEUDO="Mg.pbesol-spnl-rrkjus_psl.1.0.0.UPF"

# FHI-aims
AIMS_PATH='/home/elia/Google-Drive/google-personal/FHIaims/build'
EXE='aims.221103.mpi.x'
EXE='aims.221103.scalapack.mpi.x'

# i-PI
IPI_PATH='/home/elia/Google-Drive/google-personal/i-pi-sabia/bin'
#IPI_PATH="/home/elia/i-pi-master/bin"

PARA_PREFIX="mpirun -np 4"

#
sleep_sec="2"
```

you could add these lines to your `.bashrc` file

```bash
	#Elia Stocco
ulimit -s unlimited
alias intel="source ~/intel/oneapi/setvars.sh"
alias elia="source ~/.elia"
alias vesta="~/VESTA-gtk3/VESTA"
export FHIaims_ROOT="/home/elia/FHIaims-master"
export FHIaims_default_species_ROOT="${FHIaims_ROOT}/species_defaults/defaults_2020"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

source ~/.elia
```

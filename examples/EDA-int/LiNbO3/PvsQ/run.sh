##!/bin/bash -l

for n in "0" "1" "2" "3" "4" "5" ; do
    cmd="./raven.sh '${n}' 'scf'"
    echo ${cmd}
    eval ${cmd}

    # for gdir in "1" "2" "3" ; do
    #   cmd="./raven.sh '${n}' 'nscf' '${gdir}'"
    #   echo ${cmd}
    #   eval ${cmd}
    # done
done

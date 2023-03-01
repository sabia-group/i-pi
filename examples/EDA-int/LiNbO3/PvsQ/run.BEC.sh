##!/bin/bash -l
current_wd=$(pwd)
for n in "0" "1" "2" "3" "4" "5" ; do

  cmd="cd ${current_wd}"
  echo ${cmd}
  eval ${cmd}

  cmd="cd BEC/conf-n=${n}"
  echo ${cmd}
  eval ${cmd}

  # cmd="./raven.BEC.sh '${n}' 'scf' 'original.xyz' '' '-1' '0' "
  # echo ${cmd}
  # eval ${cmd}
  #
  # for gdir in "1" "2" "3" ; do
  #   cmd="./raven.BEC.sh '${n}' 'nscf' 'original.xyz' '${gdir}' '-1' '0' "
  #   echo ${cmd}
  #   eval ${cmd}
  # done

  cmd="cp -r outdir/LiNbO3.nk=2 outdir/orginal"
  echo ${cmd}
  eval ${cmd}

  for atom in "0" "2" "4" ; do
    for xyz in "x" "y" "z" ; do

      if [ ${atom} == "0" ]  && [ ${xyz} == "z" ] ; then
        echo "skip"
      else
        cmd="cp -r outdir/orginal outdir/LiNbO3.nk=2"
        echo ${cmd}
        eval ${cmd}

        newfile="BEC.atom=${atom}.dir=${xyz}.xyz"
        cmd="./raven.BEC.sh '${n}' 'scf' '${newfile}' '' '${atom}' '${xyz}' "
        echo ${cmd}
        eval ${cmd}

        for gdir in "1" "2" "3" ; do
          cmd="./raven.BEC.sh '${n}' 'nscf' '${newfile}' '${gdir}' '${atom}' '${xyz}' "
          echo ${cmd}
          eval ${cmd}
        done

      fi

    done
  done

done

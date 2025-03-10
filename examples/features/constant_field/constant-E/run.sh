# ln -s ../../driven_dynamics/efield-water-dynamical-Z/pswater_compiled.model .
cmd="i-pi input.xml  > i-pi.out &"
echo ${cmd}
eval ${cmd}
echo "# i-PI is running"
wait
echo "# Simulation complete"
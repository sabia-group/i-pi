#!/bin/bash -e

IPI_EXE=$(which i-pi)
python -u $IPI_EXE input.xml &> log.ipi &
sleep 5
i-pi-driver -u -h driver -p 20614 -m zundel &> log.driver &

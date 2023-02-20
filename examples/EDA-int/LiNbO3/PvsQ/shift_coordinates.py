#!/bin/bash -l
from ase.io import read,write
import numpy as np
import os
import pandas as pd
import json

# def print_cell(cell,tab="\t\t"):
#     string = tab+"{:14s} {:1s} {:^10s} {:^10s} {:^10s}".format('','','x','y','z')
#     for i in range(3):
#         string += "\n"+tab+"{:14s} {:1d} : {:>10.6f} {:>10.6f} {:>10.6f}".format('lattice vector',i+1,cell[i,0],cell[i,1],cell[i,2])
#     return string

def main():
    """main routine"""


    #print(print_cell(data.cell))
    cell = np.asarray(data.cell)

    all_shift = np.arange(-0.1,0.1,0.01)
    files = pd.DataFrame(columns=["gdir","shift","file"],index=np.arange(len(all_shift)))
    df = pd.read_csv("polarization.csv")
    k = 0

    for gdir in [1]:
        for shift in all_shift:

            if shift in list(df['shift']):
                continue

            vect_shift = cell[0,:]*shift + cell[1,:]*shift + cell[2,:]*shift

            data = read('LiNbO3.scf.original.in')
            for i in range(len(data)):
                if data.numbers[i] == :
                    data.positions[i,:] += vect_shift

            write('shifted.xyz',data,format='xyz')

            os.system("./raven.sh {:d} {:.4f}".format(gdir,shift))

            files.at[k,"gdir"]  = gdir
            files.at[k,"shift"] = shift
            with open('info.json') as info:
                j = json.load(info)
                files.at[k,"file"]  = j['output']

            k += 1

    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()

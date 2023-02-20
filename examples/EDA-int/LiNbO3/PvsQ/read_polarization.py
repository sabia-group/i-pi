#!/bin/bash -l
from ase.io import read,write
import numpy as np
import os
import pandas as pd
import json


def main():
    """main routine"""

    df = pd.DataFrame(columns=["gdir","shift","file","pol","energy"])
    unit = "e/bohr^2"
    folder = "./results"
    for file in os.listdir(folder):
        if file.endswith(".out"):

            gdir = int([ j for j in file.split(".") if 'gdir' in j ][0].split("=")[1])
            temp = file.split("=")[-1].split(".")[0:2]
            shift = float(temp[0] + "." + temp[1])

            with open(folder+"/"+file) as f:
                lines = f.readlines()
            lines.reverse()
            splitted = None
            p_found = e_found = False
            for line in lines:
                if line == "\n":
                    continue                 
                #print(line)
                if "P =" in line and unit in line:
                    pol = float(line.split()[2])
                    p_found = True
                if "!" in line and "total energy" in line:
                    energy = float(line.split()[4])
                    e_found = True

                if p_found == True and e_found == True :
                    break               
            
            df = df.append({"gdir":gdir,"shift":shift,"file":folder+"/"+file,"pol":pol,"energy":energy},ignore_index=True)  

    df.to_csv("polarization.csv",index=False)
   

    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()

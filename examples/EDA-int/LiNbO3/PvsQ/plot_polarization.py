#!/bin/bash -l
from ase.io import read,write
import numpy as np
import os
import pandas as pd
import json
import matplotlib.pyplot as plt

def fix_polarization(pol,modulus,shift=0):

    pol = np.asarray(pol)
    for i in range(1,len(pol)):

        if i in [4,10] :
            print(4)

        prev = pol[i-1]
        this = pol[i]
        
        up = abs(prev - (this + modulus) )
        no = abs(prev - this )
        dw = abs(prev - (this - modulus) )

        which = np.argmin([up,no,dw]) 
        if which == 0 :
            pol[i] = this + modulus
        elif which == 2:
            pol[i] = this - modulus
        
    return pol


def main():
    """main routine"""

    df = pd.read_csv("polarization.csv")
    colors = ['red','green','blue']
    labels = ['a','b','c']
    modulus = 0.0145280


    fig, ax1 = plt.subplots(figsize=(8,6))
    ax2 = ax1.twinx()

    for gdir,c,l in zip([1],colors,labels):
        subdf = df.loc[ df['gdir'] == gdir , :].copy()  
        subdf.sort_values(by='shift',inplace=True)  
        newpol = fix_polarization(pol=subdf['pol'],modulus=modulus,shift=0)     
        ax1.scatter(subdf['shift'], subdf['pol'], color=c,label=l)
        ax1.scatter(subdf['shift'], newpol, color="blue",label=l)
        #ax1.scatter(subdf['shift'], subdf['pol']+modulus, color=c,label=l)
        #ax1.scatter(subdf['shift'], subdf['pol']-modulus, color=c,label=l)
    
    ax2.plot(subdf['shift'], subdf['energy'], color="orange",label="energy")
        
    xmin, xmax, ymin, ymax = ax1.axis()
    ax1.hlines(0,xmin,xmax,color="black",linestyles='dotted')
    ax1.hlines(+modulus,xmin,xmax,color="black",linestyles='dotted')
    ax1.hlines(-modulus,xmin,xmax,color="black",linestyles='dotted')

    plt.xlim(xmin, xmax)
    ax1.grid(True,which='both')

    #ax1.legend(loc='lower left',title='direction')
    ax2.legend(loc='lower right')

    ax1.set_xlabel('shift $\\left[\\AA\\right]$')
    ax1.set_ylabel('polarization $\\left[e/bohr^2\\right]$')
    ax2.set_ylabel('energy $\\left[Ry\\right]$')

    plt.tight_layout()
    plt.show()
    #plt.savefig('polarization.png')   

    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()

from ase.io import read,write
import numpy as np
import copy

inverted     = read('inverted.xyz')
not_inverted = read('not-inverted.xyz')

displacement = np.asarray(inverted.positions - not_inverted.positions).reshape((1,-1))
np.savetxt('displacement.txt',displacement)

displaced = inverted.copy()
N = 5
folder = "conf"
animation = "{:s}/{:s}.xyz".format(folder,"animation")
open(animation,"w").close()
with open(animation, "a") as af:
    for n in range(N+1):
        t = float(n)/N
        # t = 0 -> positions = inverted
        # t = 1 -> positions = not_inverted
        displaced.positions = not_inverted.positions*(1-t) + t*inverted.positions
        file = "{:s}/configurations.n={:d}.xyz".format(folder,n)
        write(file,displaced,format="xyz")
        with open(file) as tempfile:
            for line in tempfile:
                af.write(line)
        


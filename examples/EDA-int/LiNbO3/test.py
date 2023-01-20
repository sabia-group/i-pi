import numpy as np

print("\n\tReading files")
rotated = np.loadtxt("rotateBEC/rotated/BEC.tensor.csv",delimiter=",")
dummy   = np.loadtxt("BEC-2/BEC.temp.csv",delimiter=",")

diff = rotated - dummy

print("\terror on dummy  :",(100 * np.sqrt(np.square(diff).sum(axis=1)) / np.sqrt(np.square(dummy).sum(axis=1))).mean()," %")
print("\terror on rotated:",(100 * np.sqrt(np.square(diff).sum(axis=1)) / np.sqrt(np.square(rotated).sum(axis=1))).mean()," %")

print("\tFinished\n")
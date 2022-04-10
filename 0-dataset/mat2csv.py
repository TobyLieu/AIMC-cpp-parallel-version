import pandas as pd
import h5py
import numpy as np

mat = h5py.File('Handwritten_fea.mat', 'r')
print(mat.keys())

for i in range(6):
    data = [mat[element[i]][:] for element in mat['X']]
    res = np.transpose(data[0])
    np.savetxt('Handwritten_fea/X/X' + str(i) + '.txt',
               res,
               delimiter=" ",
               fmt="%.12f")

Y = np.transpose(mat['Y'][:])
np.savetxt('Handwritten_fea/Y/Y.txt', Y, delimiter=",", fmt="%.12f")

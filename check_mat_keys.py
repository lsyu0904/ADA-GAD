import scipy.io
import glob
import os

mat_files = glob.glob('data/*.mat')

for mat_path in mat_files:
    print(f'检查 {mat_path} 字段:')
    mat = scipy.io.loadmat(mat_path)
    print(list(mat.keys()))
    print('-'*40) 
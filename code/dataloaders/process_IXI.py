import numpy as np
from glob import glob
from tqdm import tqdm
import os
import h5py
import nibabel as nib


def covert_h5():
    listt = glob('../../../../Thèse_Rougé_Pierre/Data/IXI-MRA/*.nii.gz')
    for item in tqdm(listt):
        
        print(item.split('/')[-1])
        
        image = nib.load(item)
        header = image.header
        image = image.get_fdata()


        item = item.replace('IXI-MRA', 'IXI-MRA-bis')
        os.makedirs(item)

if __name__ == '__main__':
    covert_h5()
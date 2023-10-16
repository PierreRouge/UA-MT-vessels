import os
import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nibabel as nib


def covert_h5():
    listt = glob('../../../../Thèse_Rougé_Pierre/Data/Bullit/raw/Images/*.nii.gz')
    for item in tqdm(listt):
        
        image = nib.load(item)
        header = image.header
        image = image.get_fdata()
        
        item = item.replace('Images', 'GT')
        
        label = nib.load(item.replace('.nii.gz', '_GT.nii.gz'))
        label_header = label.header
        label = label.get_fdata()


        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)

        item = item.replace('GT', 'dataset2')
        os.makedirs(item.replace('.nii.gz', '/'))
        f = h5py.File(item.replace('.nii.gz', '/mra_norm.h5'), 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()

if __name__ == '__main__':
    covert_h5()
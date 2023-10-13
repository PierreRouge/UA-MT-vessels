import numpy as np
from glob import glob
from tqdm import tqdm
import os
import h5py
import nibabel as nib


def covert_h5():
    listt = glob('../../../../Thèse_Rougé_Pierre/Data/IXI/IXI-MRA_modified/Images/Guys/*.nii.gz')
    for item in tqdm(listt):
        
        print(item.split('/')[-1])
        
        image = nib.load(item)
        header = image.header
        image = image.get_fdata()


        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        
        label = np.ones(image.shape)

        item = item.replace('IXI-MRA', 'IXI-MRA-ss')
        item = item.replace('.nii.gz', '/mra_norm.h5')
        os.makedirs(item.replace('/mra_norm.h5', ''))
        f = h5py.File(item, 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()

if __name__ == '__main__':
    covert_h5()
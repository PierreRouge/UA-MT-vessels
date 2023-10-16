import os
import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nibabel as nib


def covert_h5():
    listt = glob('../../../../Thèse_Rougé_Pierre/Data/Bullit/raw/Images_unlabeled/*.nii.gz')
    for item in tqdm(listt):
        
        image = nib.load(item)
        header = image.header
        image = image.get_fdata()
        
        
        label = np.ones(image.shape)
        
        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)

        item = item.replace('Images_unlabeled', 'dataset')
        os.makedirs(item.replace('.nii.gz', '/'))
        f = h5py.File(item.replace('.nii.gz', '/mra_norm.h5'), 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()
        
def read_h5():
    listt =  glob('../../../../Thèse_Rougé_Pierre/Data/Bullit/raw/dataset/*/*.h5')
    
    for item in tqdm(listt):
        print(item)
        h5f = h5py.File(item, 'r')
        print(h5f['label'])
        

if __name__ == '__main__':
    # covert_h5()
    read_h5()
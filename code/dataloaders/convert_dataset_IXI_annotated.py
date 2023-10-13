import numpy as np
from glob import glob
from tqdm import tqdm
import os
import h5py
import nibabel as nib


def covert_h5():
    listt = glob('../../../../Thèse_Rougé_Pierre/Data/IXI/IXI/IOP/Images_annotated/*.nii.gz')
    for item in tqdm(listt):
        
        print(item.split('/')[-1])
        
        image = nib.load(item)
        header = image.header
        image = image.get_fdata()
        
        item = item.replace('Images', 'GT')
        
        label = nib.load(item.replace('.nii.gz', '_GT.nii.gz'))
        label_header = label.header
        label = label.get_fdata()


        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)

        item = item.replace('IXI-MRA', 'dataset')
        item = item.replace('.nii.gz', '/mra_norm.h5')
        os.makedirs(item.replace('/mra_norm.h5', ''))
        f = h5py.File(item, 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()

if __name__ == '__main__':
    covert_h5()
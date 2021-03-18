import sys, os, glob
import pandas as pd
from scipy.io import loadmat
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def save_img_as_nifti(img, outpath, segment=False):
    assert img.ndim == 4, print(img.shape)
    if segment:
        assert img.shape[0] == 1

    for chnl_idx in range(img.shape[0]):
        itk_img = sitk.GetImageFromArray(img[chnl_idx])

        if not segment:
            out_name = outpath + "_%04.0d.nii.gz" % chnl_idx
            sitk.WriteImage(itk_img, out_name)

        else:
            out_name = outpath + ".nii.gz"
            sitk.WriteImage(itk_img, out_name)
    return

root = '/scratch/shared/beegfs/rhydian/UKBiobank'
output_path = '/scratch/shared/beegfs/rhydian/UKBiobank/nnUnet-dxa-segmentation/database/nnUNet_raw_data'
scan_lists_path = '/users/rhydian/self-supervised-project/scan_lists'
taskname = 'Task101_DXASegment'
output_path = os.path.join(output_path, taskname)

if not os.path.isdir(output_path):
    os.mkdir(output_path)

# get train and test scans names
train_scans = pd.read_csv(os.path.join(scan_lists_path,'biobank_mri_dxa_id_train.csv'))
test_scans  = pd.read_csv(os.path.join(scan_lists_path,'biobank_mri_dxa_id_test.csv'))
val_scans   = pd.read_csv(os.path.join(scan_lists_path,'biobank_mri_dxa_id_val.csv'))

train_scans= train_scans['dxa_filename'].tolist()
test_scans = test_scans['dxa_filename'].tolist()
val_scans =  val_scans['dxa_filename'].tolist()

# set up output path
subdirs =  {'train_imgs': os.path.join(output_path, 'imagesTr'),
            'test_imgs' : os.path.join(output_path, 'imagesTs'),
            'train_lbls': os.path.join(output_path, 'labelsTr'),
            'test_lbls' : os.path.join(output_path, 'labelsTs'),
            'val_lbls'  : os.path.join(output_path, 'labelsVl'),
            'val_imgs'  : os.path.join(output_path, 'imagesVl')
           }

for key in subdirs:
    if not os.path.exists(subdirs[key]):
        os.mkdir(subdirs[key])

dxa_file = 'dxa'
segmentation_file = 'dxa-segmentation3'

dxa_root = os.path.join(root, dxa_file)
segmentation_root = os.path.join(root, segmentation_file)

dxa_filenames = os.listdir(os.path.join(dxa_root,'bone'))
for set in ['val']:
    if set == 'train': scans = train_scans; images_path = subdirs['train_imgs']; segmentations_path = subdirs['train_lbls']
    elif set == 'val':   scans = val_scans; images_path = subdirs['val_imgs']; segmentations_path = subdirs['val_lbls']
    else: scans = test_scans; images_path = subdirs['test_imgs']; segmentations_path = subdirs['test_lbls']
    for scan_idx, scan in enumerate(tqdm(scans)):
        filename = scan + '.mat'
        bone_file_path = os.path.join(dxa_root, 'bone',filename)
        tissue_file_path = os.path.join(dxa_root, 'tissue', filename)
        segmentation_file_path = os.path.join(segmentation_root, filename.replace('.mat','.npy'))
        bone_scan    = loadmat(bone_file_path)['scan']
        tissue_scan  = loadmat(bone_file_path)['scan']
        segmentation = np.load(segmentation_file_path)

        # ensure scans are the same shape
        if (bone_scan.shape != tissue_scan.shape):
            print('bone scan and tissue scan different shapes; %s %s' % (bone_scan.shape, tissue_scan.shape))
            continue
        # ensure segmentation and scans are the same shape
        if (segmentation.shape[:2] != tissue_scan.shape) or (segmentation.shape[:2] != bone_scan.shape):
            print('segmentation and images are not the same shape; %s %s %s' % (segmentation.shape[:2], tissue_scan.shape, bone_scan.shape))
            continue

        scan = np.stack([bone_scan, tissue_scan],axis=0)[:,np.newaxis]
        # format segmentations
        formt_segmentation = np.zeros(segmentation.shape[:2]).astype(np.int32)
        for chnl in range(segmentation.shape[-1]):
            class_label = chnl + 1
            formt_segmentation[segmentation[:,:,chnl]==1] = class_label

        save_img_as_nifti(scan, os.path.join(images_path, 'DXA_' + str(scan_idx).zfill(5)), segment=False)
        save_img_as_nifti(formt_segmentation[None, None], os.path.join(segmentations_path, 'DXA_' + str(scan_idx).zfill(5)), segment=True)






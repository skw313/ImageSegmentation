import matplotlib.pyplot as plt
import numpy as np
import os
from pydicom import dcmread
from PIL import Image
from pydicom.data import get_testdata_file

dir_code = os.getcwd()
test_path = dir_code + '/real_test_data/DCM'
save_path = dir_code + '/real_test_data/images'

archive_data = os.listdir(test_path)

for file in archive_data:
    fpath = test_path + '/' + file
    ds = dcmread(fpath)
    im = ds.pixel_array

    # Print DICOM info
    pat_name = ds.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print(f"Patient's Name...: {display_name}")
    print(f"Patient ID.......: {ds.PatientID}")
    print(f"Modality.........: {ds.Modality}")
    print(f"Study Date.......: {ds.StudyDate}")
    print(f"Image size.......: {ds.Rows} x {ds.Columns}")
    print(f"Pixel Spacing....: {ds.PixelSpacing}")
    print()

    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()

    im = (255.0 / im.max() * (im - im.min())).astype(np.uint8)
    # Save image in temporary folders
    image_name = save_path + '/image' + str(len(os.listdir(save_path)) + 1) + '.jpg'
    image = Image.fromarray(im, mode='L')
    image.save(image_name)


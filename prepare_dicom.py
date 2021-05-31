# Preparing real MPRAGE data for the model to be tested on
# Converting DICOM images to jpg images and saving to folder

import matplotlib.pyplot as plt
import numpy as np
import os
from pydicom import dcmread
from scipy import ndimage
from PIL import Image

# Data paths
dir_code = os.getcwd()
dicom_path = dir_code + '/DCM/002_S_0782'
save_path = dir_code + '/dataset/real_test_images'
archive_data = os.listdir(dicom_path)

img_size = (256, 256)
save_img = True
slice = 75


# Function that pads image (assuming pil_img is smaller than the image size we want)
def pad(pil_img, h, w):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (w, h), 0)
    result.paste(pil_img, ((w-width) // 2, (h - height) // 2))
    return result


# Function that converts MPRAGE voxels into 2D images in axial plane
def prepare_3d_mprage():
    # Load DICOM files
    files = []
    for fname in archive_data:
        print("Loading: {}".format(fname))
        fpath = dicom_path + '/' + fname
        files.append(dcmread(fpath))

    print("Number of files: {}".format(len(files)))

    # Skip files with no Slice Location attribute (eg scout views)
    slices = []
    skip_count = 0
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            skip_count += 1

    print("Skipped {} files".format(skip_count))
    print("Number of slices: {}".format(len(slices)))

    # Sort slices by slice location
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    # Pixel aspects, assuming all slices are the same
    pix_spacing = np.array(slices[0].PixelSpacing)
    sli_thickness = slices[0].SliceThickness
    ax_aspect = pix_spacing[1]/sli_thickness
    cor_aspect = pix_spacing[1]/sli_thickness

    # Create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    # Fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    # Sagittal
    im_sag = img3d[:, :, img_shape[2] // 2]
    # Coronal
    im_cor = img3d[:, img_shape[1] // 2, :]
    # Axial
    im_ax = img3d[slice, :, :].T
    im_ax = ndimage.rotate(im_ax, 270)

    # Plot 3 orthogonal slices - Comment out
    '''
    a1 = plt.subplot(1, 3, 1)
    plt.imshow(im_sag, cmap=plt.cm.gray)
    plt.axis('off')

    a2 = plt.subplot(1, 3, 2)
    plt.imshow(im_cor, cmap=plt.cm.gray)
    a2.set_aspect(cor_aspect)
    plt.axis('off')

    a3 = plt.subplot(1, 3, 3)
    plt.imshow(im_ax, cmap=plt.cm.gray)
    a3.set_aspect(ax_aspect)
    plt.axis('off')

    plt.show()
    '''

    # Convert numpy array to PIL Image
    im = (255.0 / im_ax.max() * (im_ax - im_ax.min())).astype(np.uint8)
    image = Image.fromarray(im, mode='L')
    # Correct aspect ratio
    height = int((float(image.size[1]) * float(ax_aspect)))
    image = image.resize((image.size[0], height), Image.ANTIALIAS)
    # Pad the image to the correct size
    image = pad(image, img_size[0], img_size[1])

    # Save image as jpg
    if save_img:
        image_name = save_path + '/image' + str(len(os.listdir(save_path)) + 1) + '.jpg'
        image.save(image_name)


# Function that saves 2D DICOM images as jpegs
def prepare_2d_dicom():
    for file in archive_data:
        fpath = dicom_path + '/' + file
        ds = dcmread(fpath)
        im = ds.pixel_array

        plt.imshow(im, cmap=plt.cm.gray)
        plt.show()


        im = (255.0 / im.max() * (im - im.min())).astype(np.uint8)
        # Save image in temporary folders
        image_name = save_path + '/image' + str(len(os.listdir(save_path)) + 1) + '.jpg'
        image = Image.fromarray(im, mode='L')
        image.save(image_name)


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


if __name__ == '__main__':
    prepare_3d_mprage()


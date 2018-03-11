
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I\\O (e.g. pd.read_csv)
import random
import skimage.io
import matplotlib.pyplot as plt
import cv2

# Input data files are available in the "..\\input\\" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["dir/b", "..\\input\\"],shell=True).decode("utf8"))

# Any results you write to the current directory are saved as output.

from skimage import morphology

def read_image_labels(image_id, msk, cyc):
    # most of the content in this function is taken from 'Example Metric Implementation' kernel 
    # by 'William Cukierski'
    image_file = "..\\input\\stage1_train\\{}\\images\\{}.png".format(image_id,image_id)
    mask_file = "..\\input\\stage1_train\\{}\\masks\\*.png".format(image_id)
    image = skimage.io.imread(image_file)
    masks = skimage.io.imread_collection(mask_file).concatenate()  
    height, width, _ = image.shape
    print ("Before:", image.shape, cyc)
    if cyc > 0:
        # do clache
        grid_size = 8
        bgr = image[:,:,[2,1,0]] # flip r and b
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size,grid_size))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        image = bgr[:,:,[2,1,0]]
        
        if cyc > 1:
            # just l
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size,grid_size))
            image = clahe.apply(lab[:,:,0])

            if cyc > 2:
                if np.mean(image) > 127:
                    image = cv2.bitwise_not(image)
    if (msk > 0):
        num_masks = masks.shape[0]
        for index in range(0, num_masks):
            contour = np.logical_xor(masks[index], morphology.binary_erosion(masks[index]) )
            image[contour > 0] = 255
    print ("After: ", image.shape)
    return image


def plot_images_masks(image_ids, msk, cyc):
    plt.close('all')
    fig, ax = plt.subplots(nrows=len(image_ids)//10,ncols=10, figsize=(64,64))
    for ax_index, image_id in enumerate(image_ids):
        image = read_image_labels(image_id, msk, cyc)
        ax[ax_index//10, ax_index%10].imshow(image)
    if (msk > 0):
        if cyc > 0:
            if cyc > 1:            
                if cyc > 2:
                    fig.savefig('../output/img_clache_l_fip.png')
                else:
                    fig.savefig('../output/img_clache_l.png')
            else:
                fig.savefig('../output/img_w_clache.png')
        else:
            fig.savefig('../output/img_no_clache.png')
    else:
        fig.savefig('../output/no_mask.png')

    #plt.show()

image_ids = check_output(["dir/b", "..\\input\\stage1_train\\"],shell=True).decode("utf8").split()
print("Total Images in Training set: {}".format(len(image_ids)))
random_image_ids = random.sample(image_ids, 100)
print("Randomly Selected Images: {}, their IDs: {}".format(len(random_image_ids), random_image_ids))
plot_images_masks(random_image_ids, 0, 0)
plot_images_masks(random_image_ids, 1, 0)
# plot_images_masks(random_image_ids, 1, 1)
# plot_images_masks(random_image_ids, 1, 2)
# plot_images_masks(random_image_ids, 1, 3)
